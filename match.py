import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO


def Label_processing(sample_df):
    """标签处理"""

    def create_result_df(data_df):
        """创建结果DataFrame"""
        # 初始化结果DataFrame，包含所有原始数据
        result_df = data_df.copy()
    
        # 初始化标签列
        tag_columns = ['国家标签', '名校专家', '博士专家', '低龄留学专家', '签证能手', ]
    
        for col in tag_columns:
            result_df[col] = ''
    
        return result_df

    def process_country_tags(df):
        """处理国家标签"""
        def split_countries(x):
            if pd.isna(x):
                return ''
            countries = [country.strip() for country in str(x).split(',')]
            return ','.join(set(countries))
        
        temp_tags = pd.DataFrame(index=df.index)
        temp_tags['国家标签'] = df['签约国家'].apply(split_countries)
        return temp_tags

    def process_elite_school_tags(df):
        """处理名校专家标签"""
        # 创建临时DataFrame存储标签
        temp_tags = pd.DataFrame(index=df.index)
            
        # 根据"是否包含名校"列的值来标记名校专家
        temp_tags['名校专家'] = df['是否包含名校'].apply(
            lambda x: '名校专家' if str(x).lower() == 'yes' else ''
        )
            
        return temp_tags

    def process_education_level_tags(df):
        """处理博士专家和低龄留学专家标签"""
        # 创建临时DataFrame存储标签
        temp_tags = pd.DataFrame(index=df.index)
            
        # 处理博士专家标签
        temp_tags['博士专家'] = df['留学类别唯一'].apply(
            lambda x: '博士专家' if x == '博士/研究型硕士' else ''
        )
            
        # 处理低龄留学专家标签
        temp_tags['低龄留学专家'] = df['留学类别唯一'].apply(
            lambda x: '低龄留学专家' if x == 'k12' else ''
        )
            
        return temp_tags

    def process_visa_expert_tags(df):
        """处理签证能手标签"""
        temp_tags = pd.DataFrame(index=df.index)
        temp_tags['签证能手'] = df['办理类型'].apply(
            lambda x: '签证能手' if x == '单办签证' else ''
        )
        return temp_tags

    def Label_processing_main(df):
        """处理所有标签并生成结果"""
        # 创建结果DataFrame
        result_df = create_result_df(df)
            
        # 处理各类标签 
        country_tags = process_country_tags(df)
        elite_school_tags = process_elite_school_tags(df)
        education_tags = process_education_level_tags(df)
        visa_expert_tags = process_visa_expert_tags(df)
            
        # 将标签更新到结果DataFrame中
        for df in [country_tags, elite_school_tags, education_tags, visa_expert_tags]:
            for column in df.columns:
                result_df.loc[df.index, column] = df[column]
            
        return result_df
    
    # 调用主处理函数并返回结果
    return Label_processing_main(sample_df)


def label_merge(processed_df, merge_df):
    """标签合并"""
    # 合并标签列
    tag_columns = ['国家标签', '博士专家', '低龄留学专家', '签证能手', ]
    for col in tag_columns:
        if col in processed_df.columns:
            merge_df[col] = processed_df[col]

    # 特殊处理名校专家标签
    if '名校专家' in processed_df.columns:
        # 如果主体数据表中没有名校专家列，先创建
        if '名校专家' not in merge_df.columns:
            merge_df['名校专家'] = ''
        
        # 只在主体数据表中该列为空的情况下，才从processed_df中复制数据
        empty_mask = merge_df['名校专家'].isna() | (merge_df['名校专家'] == '')
        merge_df.loc[empty_mask, '名校专家'] = processed_df.loc[empty_mask, '名校专家']

    return merge_df

def Consultant_matching(consultant_tags_file, sample_df, merge_df):
    """顾问匹配"""
        
    # 定义标签权重
    global tag_weights
    tag_weights = {
        '绝对高频国家': 20,
        '相对高频国家': 15,
        '做过国家': 10,
        '高频专业': 10,
        '做过专业': 5,
        '名校专家': 10,
        '顶级名校猎手': 10,
        '博士专家': 10,
        '博士攻坚手': 10,
        '低龄留学专家': 10,
        '低龄留学攻坚手': 10,
        '行业经验': 20,
        '文案背景': 10,
        '业务单位所在地': 5
    }
        
    # 定义案例经验权重
    global experience_weights
    experience_weights = {
        '客户院校匹配': 100,
    }
        
    # 定义工作量权重
    global workload_weights
    workload_weights = {
        '学年负荷': 50,
        '近两周负荷': 50
    }
        
    # 定义个人意愿权重
    global personal_weights
    personal_weights = {
        '个人意愿': 50,
        '文案Flag': 50
    }
        
    # 定义评分维度权重
    global dimension_weights
    dimension_weights = {
        '标签匹配': 0.5,
        '案例经验': 0.1,
        '工作量': 0.3,
        '个人意愿': 0.1
    }

    def calculate_tag_matching_score(case, consultant):
        """计算标签匹配得分"""
        tag_score_dict = {}  # 用于存储每个标签的得分
        
        # 1. 国家标签匹配
        if '国家标签' in case and pd.notna(case['国家标签']):
            case_countries = set(case['国家标签'].split('，'))
            
            # 获取顾问的各级别国家集合
            absolute_high_freq = set(consultant['绝对高频国家'].split('，')) if pd.notna(consultant['绝对高频国家']) else set()
            relative_high_freq = set(consultant['相对高频国家'].split('，')) if pd.notna(consultant['相对高频国家']) else set()
            experienced_countries = set(consultant['做过国家'].split('，')) if pd.notna(consultant['做过国家']) else set()
            
            # 1. 先检查绝对高频国家是否完全包含目标国家
            if case_countries.issubset(absolute_high_freq):
                tag_score_dict['绝对高频国家'] = tag_weights['绝对高频国家']
            elif case_countries.issubset(absolute_high_freq.union(relative_high_freq)):
                tag_score_dict['相对高频国家'] = tag_weights['相对高频国家']
            elif case_countries.issubset(absolute_high_freq.union(relative_high_freq, experienced_countries)):
                tag_score_dict['做过国家'] = tag_weights['做过国家']
        
        # 2. 专业标签匹配
        if pd.notna(case['专业标签']) and pd.notna(consultant['高频专业']):
            case_majors = set(case['专业标签'].split('，'))
            high_freq_majors = set(consultant['高频专业'].split('，')) if pd.notna(consultant['高频专业']) else set()
            experienced_majors = set(consultant['做过专业'].split('，')) if pd.notna(consultant['做过专业']) else set()
            
            if case_majors.issubset(high_freq_majors):
                tag_score_dict['高频专业'] = tag_weights['高频专业']
            elif case_majors.issubset(high_freq_majors.union(experienced_majors)):
                tag_score_dict['做过专业'] = tag_weights['做过专业']
        
        # 3. 其他标签直接匹配
        direct_match_tags = [
            '名校专家', '顶级名校猎手', '博士专家', '博士攻坚手',
            '低龄留学专家', '低龄留学攻坚手', '行业经验', '文案背景',
            '业务单位所在地'
        ]
        
        for tag in direct_match_tags:
            if pd.notna(case[tag]) and pd.notna(consultant[tag]):
                if case[tag] == consultant[tag] and case[tag] != '':
                    tag_score_dict[tag] = tag_weights[tag]
        
        return sum(tag_score_dict.values()), tag_score_dict  # 返回总分和得分字典

    def calculate_experience_score(case, consultant):
        """客户合作经验得分"""
        total_score = 0
        
        # 检查客户院校匹配
        if pd.notna(case['客户院校匹配']) and pd.notna(consultant['客户院校匹配']):
            if case['客户院校匹配'] == consultant['客户院校匹配']:
                total_score += experience_weights['客户院校匹配']  # 100分
        
        return total_score

    def calculate_workload_score(case, consultant):
        """计算工作量得分"""
        total_score = 0
        
        # 检查学年负荷
        if pd.notna(consultant['学年负荷']):
            value = str(consultant['学年负荷']).lower()
            if value in ['是', 'true', 'yes']:
                total_score += workload_weights['学年负荷']  # 50分
        
        # 检查近两周负荷
        if pd.notna(consultant['近两周负荷']):
            value = str(consultant['近两周负荷']).lower()
            if value in ['是', 'true', 'yes']:
                total_score += workload_weights['近两周负荷']  # 50分
        
        return total_score

    def calculate_personal_score(case, consultant):
        """计算个人意愿得分"""
        total_score = 0
        
        # 检查个人意愿
        if pd.notna(consultant['个人意愿']):
            value = str(consultant['个人意愿']).lower()
            if value in ['是', 'true', 'yes']:
                total_score += personal_weights['个人意愿']  # 50分
        
        # 检查文案Flag
        if pd.notna(consultant['文案Flag']):
            value = str(consultant['文案Flag']).lower()
            if value in ['是', 'true', 'yes']:
                total_score += personal_weights['文案Flag']  # 50分
        
        return total_score

    def calculate_final_score(tag_matching_score, tag_score_dict, consultant, experience_score, workload_score, personal_score):
        """计算最终得分（包含所有维度）"""
        def count_matched_tags(tag_score_dict):
            """计算匹配上的标签数量
            Args:
                tag_score_dict: 包含各标签得分的字典
            Returns:
                int: 匹配上的标签数量
            """
            count = 0
            # 国家标签匹配上就计1次
            if any(score > 0 for tag, score in tag_score_dict.items() 
                   if tag in ['绝对高频国家', '相对高频国家', '做过国家']):
                count += 1
            
            # 专业标签匹配上就计1次
            if any(score > 0 for tag, score in tag_score_dict.items()
                   if tag in ['高频专业', '做过专业']):
                count += 1
            
            # 其他标签只要得分就计数
            other_tags = ['名校专家', '顶级名校猎手', '博士专家', '博士攻坚手', 
                         '低龄留学专家', '低龄留学攻坚手', '行业经验', '文案背景', 
                         '业务单位所在地']
            for tag in other_tags:
                if tag_score_dict.get(tag, 0) > 0:
                    count += 1
            
            return count

        def count_total_consultant_tags(consultant):
            """计算顾问的总标签数
            Args:
                consultant: 顾问信息
            Returns:
                int: 总标签数
            """
            # 默认标签数（国家、专业、行业经验、业务单位所在地）
            count = 4
            
            # 计算其他标签数
            other_tags = ['名校专家', '顶级名校猎手', '博士专家', '博士攻坚手', 
                         '低龄留学专家', '低龄留学攻坚手']
            for tag in other_tags:
                if pd.notna(consultant[tag]) and consultant[tag] != '':
                    count += 1
            
            return count
        
        # 计算标签匹配率
        matched_tags = count_matched_tags(tag_score_dict)
        total_tags = count_total_consultant_tags(consultant)
        tag_match_ratio = matched_tags / total_tags if total_tags > 0 else 0
        
        # 计算各维度最终得分
        final_tag_score = (tag_matching_score / 100) * dimension_weights['标签匹配'] * 100 * tag_match_ratio
        final_exp_score = (experience_score / 100) * dimension_weights['案例经验'] * 100
        final_workload_score = (workload_score / 100) * dimension_weights['工作量'] * 100
        final_personal_score = (personal_score / 100) * dimension_weights['个人意愿'] * 100
        
        return final_tag_score + final_exp_score + final_workload_score + final_personal_score

    def find_best_matches(consultant_tags_file, sample_df, merge_df):
        """找到每条案例得分最高的顾问们
        Args:
            consultant_tags_file: 顾问标签DataFrame
            sample_df: 原始案例数据
            merge_df: 处理后的待匹配案例数据
        Returns:
            dict: 每条案例的最佳匹配顾问列表
        """
        # 存储所有案例的匹配结果
        all_matches = {}
        
        # 对每条案例进行匹配
        for idx, case in merge_df.iterrows():
            scores = []
            
            # 计算每个顾问对当前案例的得分
            for _, consultant in consultant_tags_file.iterrows():
                # 获取标签匹配得分和得分字典
                tag_matching_score, tag_score_dict = calculate_tag_matching_score(case, consultant)
                experience_score = calculate_experience_score(case, consultant)
                workload_score = calculate_workload_score(case, consultant)
                personal_score = calculate_personal_score(case, consultant)
                
                # 计算最终得分
                final_score = calculate_final_score(
                    tag_matching_score,
                    tag_score_dict,
                    consultant,
                    experience_score,
                    workload_score,
                    personal_score
                )
                
                scores.append({
                    'name': consultant['文案顾问'],
                    'score': final_score
                })
            
            # 按得分降序排序
            scores.sort(key=lambda x: -x['score'])
            
            # 获取最高分
            highest_score = scores[0]['score']
            # 获取第三高分（如果存在）
            third_score = scores[2]['score'] if len(scores) > 2 else None
            
            # 选择得分最高的顾问们（得分大于等于第三高分的所有顾问）
            selected_consultants = [
                f"{s['name']}（{s['score']:.1f}分）" 
                for s in scores 
                if (third_score is not None and s['score'] >= third_score) or 
                   (third_score is None and s['score'] == highest_score)
            ]
            
            # 存储当前案例的匹配结果
            case_key = f"案例{idx + 1}"  # 可以根据需要修改案例的标识方式
            all_matches[case_key] = selected_consultants
        
        return all_matches
    
 # 调用 find_best_matches 函数并返回结果
    return find_best_matches(consultant_tags_file, sample_df, merge_df)

def main():
    st.title("顾问匹配系统")
    
    # 初始化 session_state
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'merged_df' not in st.session_state:
        st.session_state.merged_df = None
    
    # 文件上传区域
    with st.container():
        st.subheader("数据上传")
        uploaded_sample_data = st.file_uploader("请上传案例数据", type=['xlsx'], key='sample')
        uploaded_merge_data = st.file_uploader("请上传需要合并的主体数据表", type=['xlsx'], key='merge')
        uploaded_consultant_tags = st.file_uploader("请上传文案顾问标签汇总", type=['xlsx'], key='consultant')
        
        # 读取所有上传的文件
        if uploaded_sample_data is not None:
            sample_df = pd.read_excel(uploaded_sample_data)
            st.success("案例数据上传成功")
            
        if uploaded_merge_data is not None:
            merge_df = pd.read_excel(uploaded_merge_data)
            st.success("主体数据表上传成功")
            
        if uploaded_consultant_tags is not None:
            consultant_tags_df = pd.read_excel(uploaded_consultant_tags)
            st.success("顾问标签汇总上传成功")
    
    # 处理按钮区域
    with st.container():
        st.subheader("数据处理")
        col1, col2, col3 = st.columns(3)
        
        # 标签处理按钮
        with col1:
            if st.button("开始标签处理"):
                if uploaded_sample_data is not None:
                    try:
                        st.session_state.processed_df = Label_processing(sample_df)
                        st.success("标签处理完成！")
                        # 显示处理后的数据预览
                        st.write("处理后数据预览：")
                        st.dataframe(st.session_state.processed_df.head())
                        
                        # 添加下载按钮
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            st.session_state.processed_df.to_excel(writer, index=False, sheet_name='标签处理结果')
                        st.download_button(
                            label="下载标签处理结果",
                            data=buffer.getvalue(),
                            file_name="标签处理结果.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"标签处理出错: {str(e)}")
                else:
                    st.warning("请先上传案例数据")
        
        # 标签合并按钮
        with col2:
            if st.button("开始标签合并"):
                if st.session_state.processed_df is not None and uploaded_merge_data is not None:
                    try:
                        st.session_state.merged_df = label_merge(st.session_state.processed_df, merge_df)
                        st.success("标签合并完成！")
                        # 显示合并后的数据预览
                        st.write("合并后数据预览：")
                        st.dataframe(st.session_state.merged_df.head())
                        
                        # 添加下载按钮
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            st.session_state.merged_df.to_excel(writer, index=False, sheet_name='标签合并结果')
                        st.download_button(
                            label="下载标签合并结果",
                            data=buffer.getvalue(),
                            file_name="标签合并结果.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"标签合并出错: {str(e)}")
                else:
                    st.warning("请先完成标签处理并上传主体数据表")
        
        # 顾问匹配按钮
        with col3:
            if st.button("开始顾问匹配"):
                if st.session_state.merged_df is not None and uploaded_consultant_tags is not None:
                    try:
                        # 调用顾问匹配函数
                        matching_results = Consultant_matching(
                            consultant_tags_df,
                            sample_df,
                            st.session_state.merged_df
                        )
                        st.success("顾问匹配完成！")
                        
                        # 将匹配结果添加到原始sample数据中
                        result_df = sample_df.copy()
                        result_df['匹配文案列表'] = ''
                        
                        # 将匹配结果填入对应行
                        for case, consultants in matching_results.items():
                            idx = int(case.replace('案例', '')) - 1
                            consultant_str = '；'.join(consultants)
                            result_df.loc[idx, '匹配文案列表'] = consultant_str
                        
                        # 显示结果预览
                        st.write("匹配结果预览：")
                        st.dataframe(result_df)
                        
                        # 添加下载按钮
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            result_df.to_excel(writer, index=False, sheet_name='顾问匹配结果')
                        st.download_button(
                            label="下载顾问匹配结果",
                            data=buffer.getvalue(),
                            file_name="顾问匹配结果.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"顾问匹配出错: {str(e)}")
                else:
                    st.warning("请先完成标签合并并上传顾问标签汇总")
    
    # 显示处理状态
    with st.container():
        st.subheader("处理状态")
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.write("标签处理状态:", "✅ 完成" if st.session_state.processed_df is not None else "⏳ 待处理")
        with status_col2:
            st.write("标签合并状态:", "✅ 完成" if st.session_state.merged_df is not None else "⏳ 待处理")
        with status_col3:
            st.write("顾问匹配状态:", "✅ 完成" if 'matching_results' in locals() else "⏳ 待处理")

if __name__ == "__main__":
    main()

#cd agent
#streamlit run match.py
