import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import base64
import re
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import PatternRecognizer, Pattern
import tempfile
import os
import fitz  # PyMuPDF
from typing import List, Tuple, Dict, Any
import sys
import subprocess
import importlib.util
import requests

def load_spacy_model(model_name):
    """加载或安装指定的 spaCy 模型"""
    try:
        # 尝试导入模型
        if model_name == "zh_core_web_md":
            import zh_core_web_md
            return zh_core_web_md.load()
        elif model_name == "en_core_web_md":
            import en_core_web_md
            return en_core_web_md.load()
    except ImportError:
        # 如果导入失败，尝试从本地安装
        try:
            # 获取当前文件的目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # WHL 文件的相对路径
            whl_path = os.path.join(current_dir, "models", f"{model_name}-3.8.0-py3-none-any.whl")
            
            # 如果本地不存在，从Release下载
            if not os.path.exists(whl_path):
                whl_path = download_model_from_release(model_name)
            
            if whl_path and os.path.exists(whl_path):
                print(f"正在从本地安装 spaCy 模型: {whl_path}")
                # 使用 pip 从本地安装
                subprocess.check_call([sys.executable, "-m", "pip", "install", whl_path])
                
                # 安装后重新导入
                if model_name == "zh_core_web_md":
                    import zh_core_web_md
                    return zh_core_web_md.load()
                elif model_name == "en_core_web_md":
                    import en_core_web_md
                    return en_core_web_md.load()
            else:
                print(f"找不到本地模型文件: {whl_path}")
                return None
        except Exception as e:
            print(f"安装模型时出错: {str(e)}")
            return None

def download_model_from_release(model_name):
    """从GitHub Release下载模型"""
    release_url = f"https://github.com/hongjian03/agent2/releases/download/v1.0/{model_name}-3.8.0-py3-none-any.whl"
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", f"{model_name}-3.8.0-py3-none-any.whl")
    
    # 确保models目录存在
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # 下载文件
    print(f"从Release下载模型: {release_url}")
    response = requests.get(release_url)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"模型下载成功: {local_path}")
        return local_path
    else:
        print(f"下载失败，状态码: {response.status_code}")
        return None

class TranscriptPreprocessor:
    def __init__(self):
        # 加载spaCy中英文模型
        self.nlp_en = None
        self.nlp_zh = None
        
        try:
            # 尝试从本地加载英文模型
            self.nlp_en = load_spacy_model("en_core_web_md")
            if self.nlp_en is None:
                st.warning("无法加载英文模型，将使用备用方法")
        except Exception as e:
            st.warning(f"加载英文模型出错: {str(e)}")
        
        try:
            # 尝试从本地加载中文模型
            self.nlp_zh = load_spacy_model("zh_core_web_md")
            if self.nlp_zh is None:
                st.warning("无法加载中文模型，将使用备用方法")
        except Exception as e:
            st.warning(f"加载中文模型出错: {str(e)}")
        
        # 初始化Presidio
        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            
            # 添加自定义模式识别器
            student_id_pattern = Pattern(name="student_id_pattern", regex=r"\d{8,12}", score=0.75)
            student_id_recognizer = PatternRecognizer(supported_entity="STUDENT_ID", 
                                                     patterns=[student_id_pattern])
            
            date_patterns = [
                Pattern(name="formal_date", regex=r"\d{1,2}/[A-Za-z]{3}/\d{4}", score=0.75),
                Pattern(name="date_yyyy_mm_dd", regex=r"\d{4}-\d{1,2}-\d{1,2}", score=0.75)
            ]
            date_recognizer = PatternRecognizer(supported_entity="DATE", patterns=date_patterns)
            
            self.analyzer.registry.add_recognizer(student_id_recognizer)
            self.analyzer.registry.add_recognizer(date_recognizer)
        except Exception as e:
            st.error(f"初始化Presidio出错: {str(e)}")
            self.analyzer = None
            self.anonymizer = None
    
    def detect_sensitive_info(self, text, is_chinese=False):
        """识别文本中的敏感信息，返回敏感区域和类型"""
        sensitive_spans = []
        entity_types = []
        
        # 使用spaCy进行NER
        if is_chinese and self.nlp_zh:
            doc = self.nlp_zh(text)
        elif self.nlp_en:
            doc = self.nlp_en(text)
        else:
            # 如果模型未加载，使用正则表达式匹配
            return self.detect_sensitive_info_regex(text)
        
        # 收集人名、组织名等实体
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PER", "LOC"]:
                sensitive_spans.append((ent.start_char, ent.end_char))
                entity_types.append(ent.label_)
        
        # 使用Presidio进行敏感信息检测
        if self.analyzer:
            analyzer_results = self.analyzer.analyze(text=text,
                                                   entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
                                                            "STUDENT_ID", "DATE", "ID", "NRP", "US_SSN"])
            
            # 添加Presidio检测到的敏感信息范围
            for result in analyzer_results:
                sensitive_spans.append((result.start, result.end))
                entity_types.append(result.entity_type)
        
        return list(zip(sensitive_spans, entity_types))
    
    def detect_sensitive_info_regex(self, text):
        """使用正则表达式匹配敏感信息（备用方案）"""
        patterns = {
            "姓名": r'(?:姓名|名字|Name)[：:]\s*([^\s,;，；]{2,5})',
            "学号": r'(?:学号|Student ID|ID)[：:]*\s*([0-9A-Za-z]{5,12})',
            "出生日期": r'(?:出生日期|生日|DOB|Date of Birth)[：:]*\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]{3}[\s,]+\d{1,2}[\s,]+\d{4})',
            "电子邮件": r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            "电话": r'(?:电话|Tel|Phone|Mobile)[：:]*\s*(\+?[\d\s()-]{8,15})',
        }
        
        sensitive_spans = []
        entity_types = []
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                if match.groups():
                    # 将匹配组的位置添加到敏感范围
                    start = match.start(1)
                    end = match.end(1)
                    sensitive_spans.append((start, end))
                    entity_types.append(entity_type)
        
        return list(zip(sensitive_spans, entity_types))
    
    def process_image_with_visualization(self, image_bytes):
        """处理单个图像，可视化脱敏过程"""
        # 将字节转换为OpenCV图像
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 创建原始图像的副本用于显示
        original_img = img.copy()
        
        # 创建OCR结果图像的副本
        ocr_img = img.copy()
        
        # 使用OCR提取文本和位置信息
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ocr_result = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, lang='chi_sim+eng')
        
        # 创建一个字典存储识别过程
        visualization_data = {
            "original": cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
            "ocr_boxes": cv2.cvtColor(ocr_img.copy(), cv2.COLOR_BGR2RGB),
            "detected_sensitive": cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB),
            "final_masked": None,
            "text_blocks": [],
            "sensitive_info": []
        }
        
        # 绘制OCR识别的文本框
        for i in range(len(ocr_result['text'])):
            text = ocr_result['text'][i]
            if not text or len(text.strip()) < 2 or ocr_result['conf'][i] < 40:
                continue
                
            x = ocr_result['left'][i]
            y = ocr_result['top'][i]
            w = ocr_result['width'][i]
            h = ocr_result['height'][i]
            
            # 在OCR图像上绘制矩形和文本
            cv2.rectangle(visualization_data["ocr_boxes"], (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 保存识别的文本块用于可视化
            visualization_data["text_blocks"].append({
                "text": text,
                "confidence": ocr_result['conf'][i],
                "position": (x, y, w, h)
            })
            
            # 检测文本是否包含中文
            has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in text)
            
            # 使用spaCy和Presidio检测敏感信息
            sensitive_info = self.detect_sensitive_info(text, is_chinese=has_chinese)
            
            # 如果包含敏感信息，在图像上进行标记
            if sensitive_info:
                for (span, entity_type) in sensitive_info:
                    # 保存识别的敏感信息
                    start, end = span
                    if start < len(text) and end <= len(text):
                        sensitive_text = text[start:end]
                        visualization_data["sensitive_info"].append({
                            "text": sensitive_text,
                            "type": entity_type,
                            "position": (x, y, w, h)
                        })
                        
                        # 在敏感信息检测图像上绘制红色矩形
                        cv2.rectangle(visualization_data["detected_sensitive"], 
                                    (x, y), (x + w, y + h), (255, 0, 0), 2)
                        
                        # 添加实体类型标签
                        cv2.putText(visualization_data["detected_sensitive"], 
                                   entity_type, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
                        # 在最终图像上遮盖敏感信息
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
        
        # 针对典型成绩单格式的额外处理
        self._mask_typical_regions(img)
        
        # 保存最终遮盖后的图像
        visualization_data["final_masked"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 将处理后的图像转换回字节
        is_success, buffer = cv2.imencode(".png", img)
        if is_success:
            return buffer.tobytes(), visualization_data
        else:
            return image_bytes, visualization_data
    
    def _mask_typical_regions(self, img):
        """遮盖成绩单上典型的个人信息区域"""
        height, width = img.shape[:2]
        
        # 遮盖左上角区域 (通常包含姓名、学号等)
        top_left_mask = np.zeros((int(height * 0.15), int(width * 0.3), 3), dtype=np.uint8)
        img[:int(height * 0.15), :int(width * 0.3)] = cv2.addWeighted(
            img[:int(height * 0.15), :int(width * 0.3)], 0.3,
            top_left_mask, 0.7, 0
        )

def main():
    st.title("成绩单个人信息脱敏工具")
    
    # 初始化预处理器
    preprocessor = TranscriptPreprocessor()
    
    # 文件上传
    uploaded_files = st.file_uploader("上传成绩单图片或PDF", 
                                      type=['jpg', 'jpeg', 'png', 'pdf'], 
                                      accept_multiple_files=True)
    
    if uploaded_files:
        # 处理选项
        with st.expander("处理选项", expanded=True):
            show_visualization = st.checkbox("显示处理过程可视化", value=True)
            mask_top_left = st.checkbox("自动遮盖左上角区域(包含姓名学号)", value=True)
            ocr_threshold = st.slider("OCR置信度阈值", min_value=0, max_value=100, value=40)
        
        # 处理按钮
        if st.button("开始处理"):
            processed_files = []
            
            for file in uploaded_files:
                file_bytes = file.read()
                file_extension = file.name.split('.')[-1].lower()
                
                # 创建进度容器
                process_container = st.container()
                with process_container:
                    st.subheader(f"处理文件: {file.name}")
                    
                    if file_extension in ['jpg', 'jpeg', 'png']:
                        # 处理图片文件
                        with st.spinner("正在处理图片..."):
                            processed_bytes, visualization_data = preprocessor.process_image_with_visualization(file_bytes)
                            
                            if show_visualization:
                                # 显示处理过程
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(visualization_data["original"], caption="原始图像", use_column_width=True)
                                with col2:
                                    st.image(visualization_data["ocr_boxes"], caption="OCR文本检测", use_column_width=True)
                                
                                col3, col4 = st.columns(2)
                                with col3:
                                    st.image(visualization_data["detected_sensitive"], caption="敏感信息标记", use_column_width=True)
                                with col4:
                                    st.image(visualization_data["final_masked"], caption="最终脱敏结果", use_column_width=True)
                                
                                # 显示文本分析结果
                                with st.expander("查看检测到的文本"):
                                    for block in visualization_data["text_blocks"]:
                                        st.write(f"文本: {block['text']} (置信度: {block['confidence']})")
                                
                                with st.expander("查看检测到的敏感信息"):
                                    for info in visualization_data["sensitive_info"]:
                                        st.write(f"类型: {info['type']}, 内容: {info['text']}")
                            
                            # 显示处理前后对比
                            comparison_col1, comparison_col2 = st.columns(2)
                            with comparison_col1:
                                st.image(visualization_data["original"], caption="处理前", use_column_width=True)
                            with comparison_col2:
                                st.image(visualization_data["final_masked"], caption="处理后", use_column_width=True)
                            
                            # 提供下载链接
                            st.download_button(
                                label="下载处理后的图片",
                                data=processed_bytes,
                                file_name=f"masked_{file.name}",
                                mime=f"image/{file_extension}"
                            )
                    
                    elif file_extension == 'pdf':
                        # 处理PDF文件
                        with st.spinner("正在处理PDF..."):
                            try:
                                pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
                                st.write(f"PDF文件包含 {len(pdf_document)} 页")
                                
                                for page_num in range(len(pdf_document)):
                                    st.write(f"处理第 {page_num+1} 页...")
                                    page = pdf_document[page_num]
                                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                                    img_bytes = pix.tobytes("png")
                                    
                                    # 处理提取的每一页图像
                                    processed_bytes, visualization_data = preprocessor.process_image_with_visualization(img_bytes)
                                    
                                    if show_visualization:
                                        st.image(visualization_data["final_masked"], 
                                               caption=f"第 {page_num+1} 页 - 处理后", 
                                               use_column_width=True)
                                    
                                st.success(f"PDF文件 {file.name} 处理完成")
                                # TODO: 将处理后的页面合并回PDF
                            except Exception as e:
                                st.error(f"处理PDF文件时出错: {str(e)}")
                
                # 重置文件指针
                file.seek(0)
            
            st.success("所有文件处理完成！")

if __name__ == "__main__":
    main()
