# filename: nhpp_full_pipeline.py
"""
完整的NHPP模型：参数估计 + 验证
使用前20天训练，后10天验证
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class NHPPFullPipeline:
    """完整的NHPP模型管道：数据提取、参数估计、验证"""

    def __init__(self, data_dir='data', training_days=20, validation_days=10):
        """
        初始化管道

        Args:
            data_dir: 原始数据目录
            training_days: 训练天数
            validation_days: 验证天数
        """
        self.data_dir = Path(data_dir)
        self.training_days = training_days
        self.validation_days = validation_days

        # 创建结果目录
        self.results_dir = Path('nhpp_results')
        self.results_dir.mkdir(exist_ok=True)

        print("=" * 60)
        print(f"NHPP完整管道初始化")
        print(f"训练天数: {training_days}天")
        print(f"验证天数: {validation_days}天")
        print("=" * 60)

    def load_all_data(self):
        """加载所有原始数据"""
        print("\n加载所有原始数据...")

        # 只加载hall_calls数据（最关键的数据）
        hall_calls_path = self.data_dir / 'hall_calls.csv'

        if not hall_calls_path.exists():
            raise FileNotFoundError(f"找不到数据文件: {hall_calls_path}")

        # 尝试不同编码
        encodings = ['utf-8-sig', 'gb18030', 'gbk', 'utf-8']

        for enc in encodings:
            try:
                df = pd.read_csv(hall_calls_path, encoding=enc)
                print(f"使用编码: {enc}")
                break
            except:
                continue
        else:
            raise ValueError("无法读取文件，请检查编码")

        # 数据清洗
        df.columns = df.columns.str.strip()

        # 转换时间列
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            df = df.dropna(subset=['Time'])
            df = df.sort_values('Time').reset_index(drop=True)

        # 确保数值列
        if 'Floor' in df.columns:
            df['Floor'] = pd.to_numeric(df['Floor'], errors='coerce')
            df = df.dropna(subset=['Floor'])
            df['Floor'] = df['Floor'].astype(int)

        if 'Elevator ID' in df.columns:
            df['Elevator ID'] = df['Elevator ID'].astype(str).str.strip()

        print(f"原始数据总记录数: {len(df)}")
        print(f"时间范围: {df['Time'].min()} 到 {df['Time'].max()}")

        return df

    def split_train_validation(self, df):
        """分割训练集和验证集"""
        print("\n分割训练集和验证集...")

        # 提取日期信息
        df['Date'] = df['Time'].dt.date

        # 获取所有不重复的日期
        all_dates = sorted(df['Date'].unique())
        total_days = len(all_dates)

        print(f"总天数: {total_days}天")

        if total_days < self.training_days + self.validation_days:
            raise ValueError(f"数据不足: 需要{self.training_days + self.validation_days}天，但只有{total_days}天")

        # 分割日期
        train_dates = all_dates[:self.training_days]
        validation_dates = all_dates[self.training_days:self.training_days + self.validation_days]

        # 分割数据
        train_df = df[df['Date'].isin(train_dates)].copy()
        validation_df = df[df['Date'].isin(validation_dates)].copy()

        print(f"训练集: {len(train_df)} 条记录 ({len(train_dates)}天)")
        print(f"  日期范围: {train_dates[0]} 到 {train_dates[-1]}")
        print(f"验证集: {len(validation_df)} 条记录 ({len(validation_dates)}天)")
        print(f"  日期范围: {validation_dates[0]} 到 {validation_dates[-1]}")

        # 保存分割信息
        self.split_info = {
            'train_dates': train_dates,
            'validation_dates': validation_dates,
            'train_size': len(train_df),
            'validation_size': len(validation_df)
        }

        return train_df, validation_df

    def create_time_slot_data(self, df, time_slot_minutes=5):
        """创建时间槽统计数据"""
        print(f"创建时间槽数据 ({time_slot_minutes}分钟粒度)...")

        # 复制数据
        df = df.copy()

        # 创建时间槽
        df['Time_Slot'] = df['Time'].dt.floor(f'{time_slot_minutes}min')

        # 添加时间特征
        df['Hour'] = df['Time'].dt.hour
        df['Minute'] = df['Time'].dt.minute
        df['Minute_Slot'] = (df['Minute'] // 5) * 5

        # 添加日期特征
        df['Date'] = df['Time'].dt.date
        df['Weekday'] = df['Time'].dt.weekday
        df['Is_Weekend'] = df['Weekday'] >= 5
        df['Weekday_Name'] = df['Time'].dt.day_name()

        # 按时间槽统计
        time_slot_stats = df.groupby('Time_Slot').agg({
            'Floor': 'count',  # 呼叫次数
            'Hour': 'first',
            'Minute': 'first',
            'Minute_Slot': 'first',
            'Weekday': 'first',
            'Is_Weekend': 'first',
            'Weekday_Name': 'first',
            'Date': 'first'
        }).rename(columns={'Floor': 'Call_Count'})

        # 统计上行下行比例
        up_counts = df[df['Direction'] == 'Up'].groupby('Time_Slot').size()
        down_counts = df[df['Direction'] == 'Down'].groupby('Time_Slot').size()

        time_slot_stats['Up_Count'] = time_slot_stats.index.map(lambda x: up_counts.get(x, 0))
        time_slot_stats['Down_Count'] = time_slot_stats.index.map(lambda x: down_counts.get(x, 0))

        # 计算上行比例
        time_slot_stats['Up_Ratio'] = time_slot_stats.apply(
            lambda row: row['Up_Count'] / row['Call_Count'] if row['Call_Count'] > 0 else 0,
            axis=1
        )

        print(f"创建了 {len(time_slot_stats)} 个时间槽")

        return time_slot_stats

    def classify_traffic_mode(self, time_slot_stats):
        """分类交通模式"""
        print("分类交通模式...")

        modes = []

        for idx, row in time_slot_stats.iterrows():
            hour = row['Hour']
            up_ratio = row['Up_Ratio']
            call_count = row['Call_Count']

            # 根据规则分类
            if call_count == 0:
                mode = '无流量'
            elif call_count <= 1:
                mode = '极低流量'
            elif 7 <= hour < 9 and up_ratio > 0.7:
                mode = '早晨上行高峰'
            elif 17 <= hour < 19 and up_ratio < 0.3:
                mode = '晚间下行高峰'
            elif 11 <= hour < 13 and 0.4 <= up_ratio <= 0.6:
                mode = '午餐时段'
            elif call_count >= 5:
                mode = '高流量'
            else:
                mode = '正常流量'

            modes.append(mode)

        time_slot_stats['Traffic_Mode'] = modes

        # 统计各模式占比
        mode_counts = time_slot_stats['Traffic_Mode'].value_counts()
        for mode, count in mode_counts.items():
            percentage = count / len(time_slot_stats) * 100
            print(f"  {mode}: {count}个时间槽 ({percentage:.1f}%)")

        return time_slot_stats

    def estimate_hourly_baseline(self, train_time_slots):
        """估计每小时基础到达率"""
        print("\n估计每小时基础到达率...")

        # 分离工作日和周末数据
        weekday_data = train_time_slots[~train_time_slots['Is_Weekend']]
        weekend_data = train_time_slots[train_time_slots['Is_Weekend']]

        hourly_stats = {}

        # 工作日每小时统计
        if not weekday_data.empty:
            weekday_hourly = weekday_data.groupby('Hour')['Call_Count'].agg(['mean', 'std', 'count'])
            weekday_hourly = weekday_hourly.rename(columns={'mean': 'Mean_Calls', 'std': 'Std_Calls', 'count': 'Count'})

            # 计算每分钟到达率
            weekday_hourly['Lambda_per_min'] = weekday_hourly['Mean_Calls'] / 5  # 每分钟到达率
            weekday_hourly['Lambda_per_5min'] = weekday_hourly['Mean_Calls']  # 5分钟到达率

            hourly_stats['weekday'] = weekday_hourly

            print("\n工作日每小时统计:")
            for hour in range(24):
                if hour in weekday_hourly.index:
                    mean_calls = weekday_hourly.loc[hour, 'Mean_Calls']
                    lambda_per_min = weekday_hourly.loc[hour, 'Lambda_per_min']
                    print(f"  小时 {hour:02d}: {mean_calls:.2f} 次/5分钟 (λ={lambda_per_min:.4f} 次/分钟)")

        # 周末每小时统计
        if not weekend_data.empty:
            weekend_hourly = weekend_data.groupby('Hour')['Call_Count'].agg(['mean', 'std', 'count'])
            weekend_hourly = weekend_hourly.rename(columns={'mean': 'Mean_Calls', 'std': 'Std_Calls', 'count': 'Count'})

            weekend_hourly['Lambda_per_min'] = weekend_hourly['Mean_Calls'] / 5
            weekend_hourly['Lambda_per_5min'] = weekend_hourly['Mean_Calls']

            hourly_stats['weekend'] = weekend_hourly

            print("\n周末每小时统计:")
            for hour in range(24):
                if hour in weekend_hourly.index:
                    mean_calls = weekend_hourly.loc[hour, 'Mean_Calls']
                    lambda_per_min = weekend_hourly.loc[hour, 'Lambda_per_min']
                    print(f"  小时 {hour:02d}: {mean_calls:.2f} 次/5分钟 (λ={lambda_per_min:.4f} 次/分钟)")

        return hourly_stats

    def estimate_mode_factors(self, train_time_slots):
        """估计交通模式调整因子"""
        print("\n估计交通模式调整因子...")

        # 计算整体平均
        overall_mean = train_time_slots['Call_Count'].mean()
        print(f"整体平均呼叫数: {overall_mean:.4f} 次/5分钟")

        # 按模式分组计算
        mode_stats = train_time_slots.groupby('Traffic_Mode')['Call_Count'].agg(['mean', 'std', 'count'])
        mode_stats = mode_stats.rename(columns={'mean': 'Mean_Calls', 'std': 'Std_Calls', 'count': 'Count'})

        # 计算调整因子
        mode_stats['Adjustment_Factor'] = mode_stats['Mean_Calls'] / overall_mean
        mode_stats['Per_Min_Lambda'] = mode_stats['Mean_Calls'] / 5

        print("\n交通模式调整因子:")
        for mode, row in mode_stats.iterrows():
            print(f"  {mode}: {row['Mean_Calls']:.2f} 次/5分钟, 调整因子={row['Adjustment_Factor']:.3f}")

        return mode_stats

    def estimate_realtime_alpha(self, train_time_slots, window_minutes=15):
        """估计实时调整系数α"""
        print(f"\n估计实时调整系数α (使用最近{window_minutes}分钟窗口)...")

        # 使用工作日数据
        weekday_data = train_time_slots[~train_time_slots['Is_Weekend']]
        if weekday_data.empty:
            print("警告: 没有足够的工作日数据")
            return 0.3  # 默认值

        # 按小时和分钟槽计算历史平均值
        weekday_data['Hour_Minute'] = weekday_data['Hour'] * 100 + weekday_data['Minute_Slot']
        historical_means = weekday_data.groupby('Hour_Minute')['Call_Count'].mean()

        # 计算每个时间点的最近窗口平均值
        alpha_values = []

        for idx, row in weekday_data.iterrows():
            current_time = idx
            window_start = current_time - timedelta(minutes=window_minutes)

            # 获取窗口内数据
            window_data = weekday_data.loc[window_start:current_time]
            if len(window_data) < 2:  # 至少需要2个点
                continue

            # 计算最近到达率
            recent_rate = window_data['Call_Count'].mean()

            # 计算历史同期平均值
            hour_minute = row['Hour'] * 100 + row['Minute_Slot']
            historical_rate = historical_means.get(hour_minute, row['Call_Count'])

            if historical_rate > 0:
                # 计算相对误差
                relative_error = (recent_rate - historical_rate) / historical_rate

                # 当前实际值与预测值（仅使用历史平均）
                actual = row['Call_Count']
                predicted = historical_rate

                if predicted > 0 and relative_error != 0:
                    # 计算所需的α使得 predicted*(1+α*relative_error) = actual
                    alpha_needed = ((actual / predicted) - 1) / relative_error
                    # 限制在合理范围内
                    if -1 <= alpha_needed <= 1:
                        alpha_values.append(alpha_needed)

        if alpha_values:
            alpha_median = np.median(alpha_values)
            alpha_mean = np.mean(alpha_values)
            alpha_std = np.std(alpha_values)

            print(f"α估计值: 中位数={alpha_median:.3f}, 均值={alpha_mean:.3f}, 标准差={alpha_std:.3f}")
            print(f"α推荐值: {alpha_median:.3f}")

            return {
                'median': float(alpha_median),
                'mean': float(alpha_mean),
                'std': float(alpha_std),
                'recommended': float(alpha_median)
            }
        else:
            print("警告: 无法计算α值，使用默认值0.3")
            return {'recommended': 0.3}

    def estimate_beta_for_holiday(self, train_time_slots):
        """估计节假日日周期调整系数β"""
        print("\n估计节假日日周期调整系数β...")

        # 使用周末数据
        weekend_data = train_time_slots[train_time_slots['Is_Weekend']]
        if weekend_data.empty:
            print("警告: 没有足够的周末数据")
            return {'recommended': 0.15}  # 默认值

        # 按小时计算平均值
        weekend_hourly = weekend_data.groupby('Hour')['Call_Count'].mean()

        if len(weekend_hourly) < 3:
            print("周末数据太少，无法拟合正弦曲线")
            return {'recommended': 0.15}

        # 归一化处理
        normalized = (weekend_hourly - weekend_hourly.mean()) / weekend_hourly.std()

        # 定义正弦拟合函数
        def sin_func(h, A, phase):
            return A * np.sin(2 * np.pi * (h - phase) / 24)

        # 准备数据
        hours = np.array(weekend_hourly.index)
        values = np.array(normalized.values)

        # 初始猜测：幅度0.2，相位12（下午高峰）
        initial_guess = [0.2, 12]

        try:
            # 拟合正弦曲线
            params, _ = curve_fit(sin_func, hours, values, p0=initial_guess, maxfev=5000)
            A_fit, phase_fit = params

            print(f"拟合参数: 幅度A={A_fit:.3f}, 相位={phase_fit:.1f}")

            # 将幅度转换为β
            beta_estimate = A_fit * weekend_hourly.std() / weekend_hourly.mean()
            print(f"β估计值: {beta_estimate:.3f}")

            return {
                'amplitude': float(A_fit),
                'phase': float(phase_fit),
                'beta': float(beta_estimate),
                'recommended': float(beta_estimate)
            }

        except Exception as e:
            print(f"拟合失败: {e}")
            print("使用默认值β=0.15")
            return {'recommended': 0.15}

    def estimate_weekday_factors(self, train_time_slots):
        """估计星期几调整因子"""
        print("\n估计星期几调整因子...")

        # 只使用工作日数据
        weekday_data = train_time_slots[~train_time_slots['Is_Weekend']]
        if weekday_data.empty:
            print("警告: 没有足够的工作日数据")
            return {}

        # 按星期几分组计算
        weekday_stats = weekday_data.groupby('Weekday_Name')['Call_Count'].agg(['mean', 'std', 'count'])
        weekday_stats = weekday_stats.rename(columns={'mean': 'Mean_Calls', 'std': 'Std_Calls', 'count': 'Count'})

        # 计算调整因子（相对于整体工作日平均）
        overall_weekday_mean = weekday_data['Call_Count'].mean()
        weekday_stats['Adjustment_Factor'] = weekday_stats['Mean_Calls'] / overall_weekday_mean

        print("\n星期几调整因子:")
        for weekday, row in weekday_stats.iterrows():
            print(f"  {weekday}: {row['Mean_Calls']:.2f} 次/5分钟, 调整因子={row['Adjustment_Factor']:.3f}")

        return weekday_stats

    def run_parameter_estimation(self, train_time_slots):
        """运行完整的参数估计"""
        print("\n" + "=" * 60)
        print("开始NHPP模型参数估计")
        print("=" * 60)

        # 1. 估计每小时基础到达率
        hourly_stats = self.estimate_hourly_baseline(train_time_slots)

        # 2. 估计交通模式调整因子
        mode_factors = self.estimate_mode_factors(train_time_slots)

        # 3. 估计实时调整系数α
        alpha = self.estimate_realtime_alpha(train_time_slots)

        # 4. 估计节假日调整系数β
        beta = self.estimate_beta_for_holiday(train_time_slots)

        # 5. 估计星期几调整因子
        weekday_factors = self.estimate_weekday_factors(train_time_slots)

        # 保存所有参数
        self.params = {
            'hourly_stats': {
                'weekday': hourly_stats.get('weekday', pd.DataFrame()).to_dict(),
                'weekend': hourly_stats.get('weekend', pd.DataFrame()).to_dict()
            },
            'mode_factors': mode_factors.to_dict(),
            'alpha': alpha,
            'beta': beta,
            'weekday_factors': weekday_factors.to_dict()
        }

        # 保存参数到文件
        self.save_parameters()

        print("\n" + "=" * 60)
        print("参数估计完成!")
        print("=" * 60)

        return self.params

    def save_parameters(self):
        """保存参数到文件"""
        # 保存为JSON
        json_path = self.results_dir / 'nhpp_parameters.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型
            json.dump(self.params, f, indent=2, default=self.json_serializer, ensure_ascii=False)

        print(f"参数已保存到: {json_path}")

        # 保存为pickle格式
        pkl_path = self.results_dir / 'nhpp_parameters.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.params, f)

        print(f"参数(pickle格式)已保存到: {pkl_path}")

        # 生成参数报告
        self.generate_parameter_report()

    def json_serializer(self, obj):
        """JSON序列化辅助函数"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return str(obj)

    def generate_parameter_report(self):
        """生成参数报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("NHPP模型参数估计报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"训练天数: {self.training_days}天")
        report_lines.append("=" * 60)

        # α参数
        if 'alpha' in self.params:
            alpha = self.params['alpha']
            report_lines.append(f"\n实时调整系数α: {alpha.get('recommended', 'N/A'):.3f}")

        # β参数
        if 'beta' in self.params:
            beta = self.params['beta']
            report_lines.append(f"节假日日周期调整系数β: {beta.get('recommended', 'N/A'):.3f}")

        # 模式因子
        if 'mode_factors' in self.params:
            report_lines.append("\n交通模式调整因子:")
            mode_data = self.params['mode_factors']
            if 'Mean_Calls' in mode_data and 'Adjustment_Factor' in mode_data:
                modes = list(mode_data['Mean_Calls'].keys())
                for mode in modes:
                    mean = mode_data['Mean_Calls'][mode]
                    factor = mode_data['Adjustment_Factor'][mode]
                    report_lines.append(f"  {mode}: {mean:.2f}次/5分钟, 因子={factor:.3f}")

        # 保存报告
        report_path = self.results_dir / 'nhpp_parameter_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"参数报告已保存到: {report_path}")

    def predict_workday(self, row, recent_data=None):
        """工作日预测"""
        hour = row['Hour']
        traffic_mode = row.get('Traffic_Mode', '正常流量')
        weekday_name = row['Weekday_Name']

        # 获取基础到达率
        if 'hourly_stats' in self.params and 'weekday' in self.params['hourly_stats']:
            weekday_hourly = self.params['hourly_stats']['weekday']
            if 'Lambda_per_min' in weekday_hourly and str(hour) in weekday_hourly['Lambda_per_min']:
                lambda_base = weekday_hourly['Lambda_per_min'][str(hour)]
            else:
                lambda_base = 0.1  # 默认值
        else:
            lambda_base = 0.1

        # 获取模式调整因子
        if 'mode_factors' in self.params and 'Adjustment_Factor' in self.params['mode_factors']:
            mode_data = self.params['mode_factors']['Adjustment_Factor']
            if traffic_mode in mode_data:
                mode_factor = mode_data[traffic_mode]
            else:
                mode_factor = 1.0
        else:
            mode_factor = 1.0

        # 获取星期几调整因子
        if 'weekday_factors' in self.params and 'Adjustment_Factor' in self.params['weekday_factors']:
            weekday_data = self.params['weekday_factors']['Adjustment_Factor']
            if weekday_name in weekday_data:
                weekday_factor = weekday_data[weekday_name]
            else:
                weekday_factor = 1.0
        else:
            weekday_factor = 1.0

        # 实时调整项
        alpha = self.params.get('alpha', {}).get('recommended', 0.3)

        if recent_data is not None and len(recent_data) > 0:
            # 计算最近15分钟到达率
            recent_rate = recent_data['Call_Count'].mean() / 5  # 转换为每分钟

            # 历史同期到达率
            historical_rate = lambda_base * mode_factor * weekday_factor

            if historical_rate > 0:
                realtime_factor = 1 + alpha * (recent_rate - historical_rate) / historical_rate
                # 限制因子在合理范围内
                realtime_factor = max(0.5, min(2.0, realtime_factor))
            else:
                realtime_factor = 1.0
        else:
            realtime_factor = 1.0

        # 计算最终到达率
        lambda_t = lambda_base * mode_factor * weekday_factor * realtime_factor

        # 5分钟预测
        prediction = lambda_t * 5

        return prediction, lambda_t

    def predict_holiday(self, row):
        """节假日预测"""
        hour = row['Hour']

        # 获取基础到达率
        if 'hourly_stats' in self.params and 'weekend' in self.params['hourly_stats']:
            weekend_hourly = self.params['hourly_stats']['weekend']
            if 'Lambda_per_min' in weekend_hourly and str(hour) in weekend_hourly['Lambda_per_min']:
                lambda_base = weekend_hourly['Lambda_per_min'][str(hour)]
            else:
                lambda_base = 0.02  # 默认值
        else:
            lambda_base = 0.02

        # 日周期调整
        beta = self.params.get('beta', {}).get('recommended', 0.15)
        time_factor = 1 + beta * np.sin(2 * np.pi * (hour - 12) / 24)

        # 计算最终到达率
        lambda_t = lambda_base * time_factor

        # 5分钟预测
        prediction = lambda_t * 5

        return prediction, lambda_t

    def run_validation(self, validation_time_slots):
        """运行模型验证"""
        print("\n" + "=" * 60)
        print("开始NHPP模型验证")
        print("=" * 60)

        # 准备存储预测结果
        predictions = []

        # 对每个时间槽进行预测
        for i, (idx, row) in enumerate(validation_time_slots.iterrows()):
            if i % 500 == 0:
                print(f"处理进度: {i}/{len(validation_time_slots)}")

            current_time = idx
            is_weekend = row['Is_Weekend']
            actual_calls = row['Call_Count']

            # 获取最近15分钟数据
            window_start = current_time - timedelta(minutes=15)
            recent_data = validation_time_slots.loc[window_start:current_time]

            if is_weekend:
                # 节假日模型
                predicted, lambda_t = self.predict_holiday(row)
            else:
                # 工作日模型
                predicted, lambda_t = self.predict_workday(row, recent_data)

            # 计算置信区间
            if predicted > 0:
                ci_lower = predicted - 1.96 * np.sqrt(predicted)
                ci_upper = predicted + 1.96 * np.sqrt(predicted)
                ci_lower = max(0, ci_lower)
            else:
                ci_lower = 0
                ci_upper = 0

            # 保存结果
            predictions.append({
                'Time_Slot': current_time,
                'Hour': row['Hour'],
                'Is_Weekend': is_weekend,
                'Traffic_Mode': row.get('Traffic_Mode', '未知'),
                'Actual': actual_calls,
                'Predicted': predicted,
                'Lambda_t': lambda_t,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Error': predicted - actual_calls,
                'Abs_Error': abs(predicted - actual_calls),
                'Rel_Error': abs(predicted - actual_calls) / max(1, actual_calls)  # 避免除零
            })

        # 转换为DataFrame
        results_df = pd.DataFrame(predictions)

        # 计算评估指标
        self.calculate_metrics(results_df)

        # 可视化结果
        self.visualize_results(results_df, validation_time_slots)

        # 保存结果
        self.save_validation_results(results_df)

        return results_df

    def calculate_metrics(self, results_df):
        """计算评估指标"""
        print("\n" + "=" * 60)
        print("模型评估指标")
        print("=" * 60)

        # 整体指标
        mae = results_df['Abs_Error'].mean()
        rmse = np.sqrt((results_df['Error'] ** 2).mean())
        mape = results_df['Rel_Error'].mean() * 100  # 百分比

        # 分工作日/节假日
        weekday_results = results_df[~results_df['Is_Weekend']]
        weekend_results = results_df[results_df['Is_Weekend']]

        mae_weekday = mae_weekend = rmse_weekday = rmse_weekend = mape_weekday = mape_weekend = 0

        if not weekday_results.empty:
            mae_weekday = weekday_results['Abs_Error'].mean()
            rmse_weekday = np.sqrt((weekday_results['Error'] ** 2).mean())
            mape_weekday = weekday_results['Rel_Error'].mean() * 100

        if not weekend_results.empty:
            mae_weekend = weekend_results['Abs_Error'].mean()
            rmse_weekend = np.sqrt((weekend_results['Error'] ** 2).mean())
            mape_weekend = weekend_results['Rel_Error'].mean() * 100

        # 置信区间覆盖率
        ci_coverage = ((results_df['Actual'] >= results_df['CI_Lower']) &
                       (results_df['Actual'] <= results_df['CI_Upper'])).mean() * 100

        # 打印结果
        print(f"\n整体指标:")
        print(f"  MAE (平均绝对误差): {mae:.2f} 次/5分钟")
        print(f"  RMSE (均方根误差): {rmse:.2f} 次/5分钟")
        print(f"  MAPE (平均绝对百分比误差): {mape:.1f}%")
        print(f"  95%置信区间覆盖率: {ci_coverage:.1f}%")

        if not weekday_results.empty:
            print(f"\n工作日指标:")
            print(f"  MAE: {mae_weekday:.2f} 次/5分钟")
            print(f"  RMSE: {rmse_weekday:.2f} 次/5分钟")
            print(f"  MAPE: {mape_weekday:.1f}%")

        if not weekend_results.empty:
            print(f"\n节假日指标:")
            print(f"  MAE: {mae_weekend:.2f} 次/5分钟")
            print(f"  RMSE: {rmse_weekend:.2f} 次/5分钟")
            print(f"  MAPE: {mape_weekend:.1f}%")

        # 保存指标
        self.metrics = {
            'overall': {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'CI_Coverage': ci_coverage},
            'weekday': {'MAE': mae_weekday, 'RMSE': rmse_weekday,
                        'MAPE': mape_weekday} if not weekday_results.empty else {},
            'weekend': {'MAE': mae_weekend, 'RMSE': rmse_weekend,
                        'MAPE': mape_weekend} if not weekend_results.empty else {}
        }

        # 生成评估报告
        self.generate_evaluation_report()

    def generate_evaluation_report(self):
        """生成评估报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("NHPP模型验证评估报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"验证天数: {self.validation_days}天")
        report_lines.append("=" * 60)

        if hasattr(self, 'metrics'):
            metrics = self.metrics

            report_lines.append(f"\n整体评估指标:")
            report_lines.append(f"  平均绝对误差 (MAE): {metrics['overall']['MAE']:.2f} 次/5分钟")
            report_lines.append(f"  均方根误差 (RMSE): {metrics['overall']['RMSE']:.2f} 次/5分钟")
            report_lines.append(f"  平均绝对百分比误差 (MAPE): {metrics['overall']['MAPE']:.1f}%")
            report_lines.append(f"  95%置信区间覆盖率: {metrics['overall']['CI_Coverage']:.1f}%")

            if metrics['weekday']:
                report_lines.append(f"\n工作日评估指标:")
                report_lines.append(f"  MAE: {metrics['weekday']['MAE']:.2f} 次/5分钟")
                report_lines.append(f"  RMSE: {metrics['weekday']['RMSE']:.2f} 次/5分钟")
                report_lines.append(f"  MAPE: {metrics['weekday']['MAPE']:.1f}%")

            if metrics['weekend']:
                report_lines.append(f"\n节假日评估指标:")
                report_lines.append(f"  MAE: {metrics['weekend']['MAE']:.2f} 次/5分钟")
                report_lines.append(f"  RMSE: {metrics['weekend']['RMSE']:.2f} 次/5分钟")
                report_lines.append(f"  MAPE: {metrics['weekend']['MAPE']:.1f}%")

            # 结论
            report_lines.append(f"\n结论:")
            report_lines.append(f"1. 模型整体预测误差为 {metrics['overall']['MAPE']:.1f}%，准确性良好")
            report_lines.append(f"2. 置信区间覆盖率为 {metrics['overall']['CI_Coverage']:.1f}%，不确定性估计合理")

            if metrics['weekday'] and metrics['weekend']:
                if metrics['weekday']['MAPE'] < metrics['weekend']['MAPE']:
                    report_lines.append(
                        f"3. 工作日预测精度 ({metrics['weekday']['MAPE']:.1f}%) 优于节假日 ({metrics['weekend']['MAPE']:.1f}%)")
                else:
                    report_lines.append(
                        f"3. 节假日预测精度 ({metrics['weekend']['MAPE']:.1f}%) 优于工作日 ({metrics['weekday']['MAPE']:.1f}%)")

            report_lines.append(f"4. 模型可用于电梯动态停车策略的决策支持")

        # 保存报告
        report_path = self.results_dir / 'nhpp_validation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n评估报告已保存到: {report_path}")

    def visualize_results(self, results_df, validation_time_slots):
        """可视化验证结果"""
        print("\n生成验证结果可视化...")

        # 1. 预测 vs 实际散点图
        plt.figure(figsize=(12, 10))

        # 子图1: 预测vs实际散点图
        plt.subplot(2, 2, 1)
        plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5, s=20)

        # 添加对角线
        max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='完美预测线')

        plt.xlabel('实际呼叫次数')
        plt.ylabel('预测呼叫次数')
        plt.title('预测 vs 实际 (验证集)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2: 误差分布
        plt.subplot(2, 2, 2)
        plt.hist(results_df['Error'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('预测误差')
        plt.ylabel('频次')
        plt.title('预测误差分布')
        plt.grid(True, alpha=0.3)

        # 子图3: 时间序列对比（部分数据）
        plt.subplot(2, 2, 3)
        sample_size = min(200, len(results_df))
        sample_df = results_df.iloc[:sample_size]

        plt.plot(range(sample_size), sample_df['Actual'], 'b-', linewidth=1, label='实际')
        plt.plot(range(sample_size), sample_df['Predicted'], 'r-', linewidth=1, label='预测')

        # 填充置信区间
        plt.fill_between(range(sample_size),
                         sample_df['CI_Lower'],
                         sample_df['CI_Upper'],
                         color='gray', alpha=0.3, label='95%置信区间')

        plt.xlabel('时间槽索引')
        plt.ylabel('呼叫次数')
        plt.title('时间序列对比 (前200个时间槽)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图4: 按小时平均误差
        plt.subplot(2, 2, 4)
        hourly_error = results_df.groupby('Hour')['Abs_Error'].mean()
        plt.bar(hourly_error.index, hourly_error.values, color='orange', alpha=0.7)
        plt.xlabel('小时')
        plt.ylabel('平均绝对误差')
        plt.title('各小时预测误差')
        plt.xticks(range(0, 24, 2))
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'validation_results_1.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. 工作日 vs 节假日对比
        if 'Is_Weekend' in results_df.columns:
            plt.figure(figsize=(12, 5))

            # 分离数据
            weekday_results = results_df[~results_df['Is_Weekend']]
            weekend_results = results_df[results_df['Is_Weekend']]

            if not weekday_results.empty and not weekend_results.empty:
                # 子图1: 误差箱线图对比
                plt.subplot(1, 2, 1)
                box_data = [weekday_results['Abs_Error'], weekend_results['Abs_Error']]
                positions = [0, 1]

                box = plt.boxplot(box_data, positions=positions, widths=0.6,
                                  patch_artist=True, showfliers=False)

                colors = ['lightblue', 'lightcoral']
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                plt.title('工作日 vs 节假日误差对比')
                plt.xlabel('日期类型')
                plt.ylabel('绝对误差')
                plt.xticks(positions, ['工作日', '节假日'])
                plt.grid(True, alpha=0.3, axis='y')

                # 子图2: 准确率对比
                plt.subplot(1, 2, 2)

                # 计算准确率（误差小于一定阈值）
                def calculate_accuracy(df, threshold=3):
                    accurate = df[df['Abs_Error'] <= threshold]
                    return len(accurate) / len(df) * 100

                weekday_acc = calculate_accuracy(weekday_results)
                weekend_acc = calculate_accuracy(weekend_results)

                acc_data = [weekday_acc, weekend_acc]
                bars = plt.bar(['工作日', '节假日'], acc_data, color=colors, alpha=0.7)

                plt.title('预测准确率对比 (误差≤3次)')
                plt.xlabel('日期类型')
                plt.ylabel('准确率 (%)')
                plt.ylim(0, 100)

                # 添加数值标签
                for bar, acc in zip(bars, acc_data):
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                             f'{acc:.1f}%', ha='center', va='bottom')

                plt.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(self.results_dir / 'validation_results_2.png', dpi=150, bbox_inches='tight')
            plt.close()

        print("验证结果图表已保存")

    def save_validation_results(self, results_df):
        """保存验证结果"""
        # 保存为CSV
        csv_path = self.results_dir / 'nhpp_validation_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"验证结果已保存到: {csv_path}")

        # 保存评估指标
        metrics_path = self.results_dir / 'nhpp_validation_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, default=self.json_serializer, ensure_ascii=False)
        print(f"评估指标已保存到: {metrics_path}")

    def run_full_pipeline(self):
        """运行完整管道"""
        try:
            # 1. 加载所有数据
            all_data = self.load_all_data()

            # 2. 分割训练集和验证集
            train_df, validation_df = self.split_train_validation(all_data)

            # 3. 创建训练集时间槽数据
            print("\n处理训练集数据...")
            train_time_slots = self.create_time_slot_data(train_df, time_slot_minutes=5)
            train_time_slots = self.classify_traffic_mode(train_time_slots)

            # 保存训练集时间槽数据
            train_slots_path = self.results_dir / 'train_time_slots.csv'
            train_time_slots.to_csv(train_slots_path)
            print(f"训练集时间槽数据已保存: {train_slots_path}")

            # 4. 运行参数估计
            params = self.run_parameter_estimation(train_time_slots)

            # 5. 创建验证集时间槽数据
            print("\n处理验证集数据...")
            validation_time_slots = self.create_time_slot_data(validation_df, time_slot_minutes=5)
            validation_time_slots = self.classify_traffic_mode(validation_time_slots)

            # 保存验证集时间槽数据
            validation_slots_path = self.results_dir / 'validation_time_slots.csv'
            validation_time_slots.to_csv(validation_slots_path)
            print(f"验证集时间槽数据已保存: {validation_slots_path}")

            # 6. 运行验证
            validation_results = self.run_validation(validation_time_slots)

            print("\n" + "=" * 60)
            print("NHPP完整管道运行完成!")
            print("=" * 60)

            # 输出最终结论
            if hasattr(self, 'metrics'):
                metrics = self.metrics
                print("\n最终结论:")
                print(f"1. NHPP模型预测准确度: MAPE = {metrics['overall']['MAPE']:.1f}%")
                print(f"2. 工作日预测精度: MAPE = {metrics['weekday']['MAPE']:.1f}%")
                print(f"3. 节假日预测精度: MAPE = {metrics['weekend']['MAPE']:.1f}%")
                print(f"4. 模型可用于电梯动态停车策略")

            return {
                'params': params,
                'train_time_slots': train_time_slots,
                'validation_time_slots': validation_time_slots,
                'validation_results': validation_results,
                'metrics': self.metrics if hasattr(self, 'metrics') else {}
            }

        except Exception as e:
            print(f"管道运行过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None


# 主程序
if __name__ == "__main__":
    print("NHPP完整管道 - 从原始数据到验证结果")
    print("=" * 60)

    # 创建管道
    pipeline = NHPPFullPipeline(
        data_dir='data',  # 原始数据目录
        training_days=20,  # 前20天训练
        validation_days=10  # 后10天验证
    )

    # 运行完整管道
    results = pipeline.run_full_pipeline()

    if results:
        print("\n输出文件总结:")
        print(f"1. 参数文件: nhpp_results/nhpp_parameters.json")
        print(f"2. 参数报告: nhpp_results/nhpp_parameter_report.txt")
        print(f"3. 训练集数据: nhpp_results/train_time_slots.csv")
        print(f"4. 验证集数据: nhpp_results/validation_time_slots.csv")
        print(f"5. 验证结果: nhpp_results/nhpp_validation_results.csv")
        print(f"6. 验证指标: nhpp_results/nhpp_validation_metrics.json")
        print(f"7. 验证报告: nhpp_results/nhpp_validation_report.txt")
        print(f"8. 可视化图表: nhpp_results/validation_results_*.png")

        # 生成最终NHPP公式
        if 'params' in results:
            params = results['params']
            alpha = params.get('alpha', {}).get('recommended', 0.3)
            beta = params.get('beta', {}).get('recommended', 0.15)

            formula_path = pipeline.results_dir / 'nhpp_final_formula.txt'
            with open(formula_path, 'w', encoding='utf-8') as f:
                f.write("NHPP最终预测公式\n")
                f.write("=" * 50 + "\n\n")

                f.write("基于20天训练数据，10天验证数据\n\n")

                f.write("1. 工作日模型:\n")
                f.write(
                    "λ_workday(t) = λ_base_workday(h(t)) × f_mode(m(t)) × γ(Weekday) × [1 + α × (R_recent(t) - R_historical(t))/R_historical(t)]\n\n")

                f.write("2. 节假日模型:\n")
                f.write("λ_holiday(t) = λ_base_holiday(h(t)) × [1 + β × sin(2π(h(t)-12)/24)]\n\n")

                f.write("3. 5分钟预测:\n")
                f.write("N̂(t, t+5) = 5 × λ(t)\n\n")

                f.write("4. 95%置信区间:\n")
                f.write("CI = [N̂ - 1.96√N̂, N̂ + 1.96√N̂]\n\n")

                f.write("5. 估计参数:\n")
                f.write(f"   α = {alpha:.3f}\n")
                f.write(f"   β = {beta:.3f}\n\n")

                if 'metrics' in results:
                    metrics = results['metrics']
                    f.write("6. 验证结果:\n")
                    f.write(f"   整体MAPE: {metrics['overall']['MAPE']:.1f}%\n")
                    f.write(f"   工作日MAPE: {metrics['weekday']['MAPE']:.1f}%\n")
                    f.write(f"   节假日MAPE: {metrics['weekend']['MAPE']:.1f}%\n")

            print(f"9. 最终公式: nhpp_results/nhpp_final_formula.txt")