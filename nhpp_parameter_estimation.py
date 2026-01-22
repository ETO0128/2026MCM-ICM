# filename: nhpp_parameter_estimation.py
"""
NHPP模型参数估计脚本
基于前20天训练数据计算模型参数
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

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class NHPPParameterEstimator:
    """NHPP模型参数估计器"""

    def __init__(self, training_data_path='training_results/time_slot_stats_training.csv'):
        """
        初始化参数估计器

        Args:
            training_data_path: 训练数据路径
        """
        print("初始化NHPP参数估计器...")
        self.training_data_path = Path(training_data_path)
        self.params = {}
        self.load_training_data()

    def load_training_data(self):
        """加载训练数据"""
        print(f"加载训练数据: {self.training_data_path}")
        if not self.training_data_path.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {self.training_data_path}")

        self.training_data = pd.read_csv(self.training_data_path, parse_dates=['Time_Slot'], index_col='Time_Slot')
        print(f"训练数据形状: {self.training_data.shape}")
        print(f"时间范围: {self.training_data.index.min()} 到 {self.training_data.index.max()}")

        # 添加额外的时间特征
        self.training_data['Hour'] = self.training_data.index.hour
        self.training_data['Minute'] = self.training_data.index.minute
        self.training_data['Minute_Slot'] = (self.training_data['Minute'] // 5) * 5

        # 确保有Is_Weekend列
        if 'Is_Weekend' not in self.training_data.columns:
            self.training_data['Weekday'] = self.training_data.index.weekday
            self.training_data['Is_Weekend'] = self.training_data['Weekday'] >= 5

    def estimate_hourly_baseline(self):
        """估计每小时基础到达率"""
        print("\n估计每小时基础到达率...")

        # 分离工作日和周末数据
        weekday_data = self.training_data[~self.training_data['Is_Weekend']]
        weekend_data = self.training_data[self.training_data['Is_Weekend']]

        # 工作日每小时统计
        if not weekday_data.empty:
            weekday_hourly = weekday_data.groupby('Hour')['Call_Count'].agg(['mean', 'std', 'count'])
            weekday_hourly = weekday_hourly.rename(columns={'mean': 'Mean_Calls', 'std': 'Std_Calls', 'count': 'Count'})

            # 计算每分钟到达率（将5分钟呼叫数转换为每分钟）
            weekday_hourly['Lambda_per_min'] = weekday_hourly['Mean_Calls'] / 5  # 每分钟到达率
            weekday_hourly['Lambda_per_5min'] = weekday_hourly['Mean_Calls']  # 5分钟到达率

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

            print("\n周末每小时统计:")
            for hour in range(24):
                if hour in weekend_hourly.index:
                    mean_calls = weekend_hourly.loc[hour, 'Mean_Calls']
                    lambda_per_min = weekend_hourly.loc[hour, 'Lambda_per_min']
                    print(f"  小时 {hour:02d}: {mean_calls:.2f} 次/5分钟 (λ={lambda_per_min:.4f} 次/分钟)")

        # 保存参数
        self.params['weekday_hourly'] = weekday_hourly.to_dict() if not weekday_data.empty else {}
        self.params['weekend_hourly'] = weekend_hourly.to_dict() if not weekend_data.empty else {}

        # 可视化
        self.plot_hourly_baseline(weekday_hourly, weekend_hourly)

        return weekday_hourly, weekend_hourly

    def estimate_mode_factors(self):
        """估计交通模式调整因子"""
        print("\n估计交通模式调整因子...")

        if 'Traffic_Mode' not in self.training_data.columns:
            print("警告: 数据中没有Traffic_Mode列")
            return {}

        # 计算整体平均
        overall_mean = self.training_data['Call_Count'].mean()
        print(f"整体平均呼叫数: {overall_mean:.4f} 次/5分钟")

        # 按模式分组计算
        mode_stats = self.training_data.groupby('Traffic_Mode')['Call_Count'].agg(['mean', 'std', 'count'])
        mode_stats = mode_stats.rename(columns={'mean': 'Mean_Calls', 'std': 'Std_Calls', 'count': 'Count'})

        # 计算调整因子
        mode_stats['Adjustment_Factor'] = mode_stats['Mean_Calls'] / overall_mean
        mode_stats['Per_Min_Lambda'] = mode_stats['Mean_Calls'] / 5

        print("\n交通模式调整因子:")
        for mode, row in mode_stats.iterrows():
            print(f"  {mode}: {row['Mean_Calls']:.2f} 次/5分钟, 调整因子={row['Adjustment_Factor']:.3f}")

        # 保存参数
        self.params['mode_factors'] = mode_stats.to_dict()

        # 可视化
        self.plot_mode_factors(mode_stats)

        return mode_stats

    def estimate_realtime_alpha(self, window_minutes=15):
        """估计实时调整系数α"""
        print(f"\n估计实时调整系数α (使用最近{window_minutes}分钟窗口)...")

        # 使用工作日数据
        weekday_data = self.training_data[~self.training_data['Is_Weekend']]
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

                if predicted > 0:
                    # 计算所需的α使得 predicted*(1+α*relative_error) = actual
                    if relative_error != 0:
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

            self.params['alpha'] = {
                'median': float(alpha_median),
                'mean': float(alpha_mean),
                'std': float(alpha_std),
                'recommended': float(alpha_median)
            }

            # 可视化α值分布
            self.plot_alpha_distribution(alpha_values, alpha_median)

            return alpha_median
        else:
            print("警告: 无法计算α值，使用默认值0.3")
            self.params['alpha'] = {'recommended': 0.3}
            return 0.3

    def estimate_beta_for_holiday(self):
        """估计节假日日周期调整系数β"""
        print("\n估计节假日日周期调整系数β...")

        # 使用周末数据
        weekend_data = self.training_data[self.training_data['Is_Weekend']]
        if weekend_data.empty:
            print("警告: 没有足够的周末数据")
            return 0.15  # 默认值

        # 按小时计算平均值
        weekend_hourly = weekend_data.groupby('Hour')['Call_Count'].mean()

        # 归一化处理
        max_rate = weekend_hourly.max()
        min_rate = weekend_hourly.min()
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
            params, covariance = curve_fit(sin_func, hours, values, p0=initial_guess)
            A_fit, phase_fit = params

            print(f"拟合参数: 幅度A={A_fit:.3f}, 相位={phase_fit:.1f}")

            # 将幅度转换为β（因为我们的模型是 1 + β * sin(...)）
            # 归一化幅度的原始尺度
            beta_estimate = A_fit * weekend_hourly.std() / weekend_hourly.mean()
            print(f"β估计值: {beta_estimate:.3f}")

            # 可视化拟合结果
            self.plot_holiday_sin_fit(hours, values, sin_func, params, weekend_hourly)

            self.params['beta'] = {
                'amplitude': float(A_fit),
                'phase': float(phase_fit),
                'beta': float(beta_estimate),
                'recommended': float(beta_estimate)
            }

            return beta_estimate

        except Exception as e:
            print(f"拟合失败: {e}")
            print("使用默认值β=0.15")
            self.params['beta'] = {'recommended': 0.15}
            return 0.15

    def estimate_weekday_factors(self):
        """估计星期几调整因子"""
        print("\n估计星期几调整因子...")

        # 添加星期几信息
        self.training_data['Weekday'] = self.training_data.index.weekday
        self.training_data['Weekday_Name'] = self.training_data.index.day_name()

        # 只使用工作日数据
        weekday_data = self.training_data[~self.training_data['Is_Weekend']]
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

        # 保存参数
        self.params['weekday_factors'] = weekday_stats.to_dict()

        # 可视化
        self.plot_weekday_factors(weekday_stats)

        return weekday_stats

    def save_parameters(self, output_path='nhpp_parameters.json'):
        """保存参数到文件"""
        output_path = Path(output_path)

        # 转换为可序列化的格式
        serializable_params = {}
        for key, value in self.params.items():
            if isinstance(value, dict):
                serializable_params[key] = value
            else:
                serializable_params[key] = str(value)

        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_params, f, indent=2, ensure_ascii=False)

        print(f"\n参数已保存到: {output_path}")

        # 同时保存为pickle格式（保留原始数据类型）
        pickle_path = output_path.with_suffix('.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.params, f)

        print(f"参数（pickle格式）已保存到: {pickle_path}")

        # 生成参数报告
        self.generate_parameter_report()

    def generate_parameter_report(self):
        """生成参数报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("NHPP模型参数估计报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        report_path = 'nhpp_parameter_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"参数报告已保存到: {report_path}")

    # 可视化方法
    def plot_hourly_baseline(self, weekday_hourly, weekend_hourly):
        """可视化每小时基础到达率"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 工作日
        if not weekday_hourly.empty:
            ax1 = axes[0]
            hours = weekday_hourly.index
            rates = weekday_hourly['Lambda_per_min']
            ax1.bar(hours, rates, color='steelblue', alpha=0.7)
            ax1.set_title('工作日每小时基础到达率', fontsize=12)
            ax1.set_xlabel('小时')
            ax1.set_ylabel('到达率 (次/分钟)')
            ax1.set_xticks(range(0, 24, 2))
            ax1.grid(True, alpha=0.3)

        # 周末
        if not weekend_hourly.empty:
            ax2 = axes[1]
            hours = weekend_hourly.index
            rates = weekend_hourly['Lambda_per_min']
            ax2.bar(hours, rates, color='lightcoral', alpha=0.7)
            ax2.set_title('周末每小时基础到达率', fontsize=12)
            ax2.set_xlabel('小时')
            ax2.set_ylabel('到达率 (次/分钟)')
            ax2.set_xticks(range(0, 24, 2))
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('hourly_baseline.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("每小时基础到达率图表已保存")

    def plot_mode_factors(self, mode_stats):
        """可视化交通模式调整因子"""
        plt.figure(figsize=(10, 6))

        modes = mode_stats.index
        factors = mode_stats['Adjustment_Factor'].values

        # 按调整因子排序
        sorted_idx = np.argsort(factors)[::-1]  # 降序
        modes_sorted = [modes[i] for i in sorted_idx]
        factors_sorted = factors[sorted_idx]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(modes)))

        bars = plt.bar(range(len(modes)), factors_sorted, color=colors, alpha=0.7)
        plt.title('交通模式调整因子', fontsize=12)
        plt.xlabel('交通模式')
        plt.ylabel('调整因子')
        plt.xticks(range(len(modes)), modes_sorted, rotation=45, ha='right')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='基准线 (1.0)')

        # 添加数值标签
        for i, v in enumerate(factors_sorted):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

        plt.legend()
        plt.tight_layout()
        plt.savefig('mode_factors.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("交通模式调整因子图表已保存")

    def plot_alpha_distribution(self, alpha_values, alpha_median):
        """可视化α值分布"""
        plt.figure(figsize=(10, 6))

        plt.hist(alpha_values, bins=30, color='lightblue', edgecolor='black', alpha=0.7)
        plt.axvline(alpha_median, color='red', linestyle='--', linewidth=2,
                    label=f'中位数: {alpha_median:.3f}')

        plt.title('实时调整系数α分布', fontsize=12)
        plt.xlabel('α值')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('alpha_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("α值分布图表已保存")

    def plot_holiday_sin_fit(self, hours, values, sin_func, params, weekend_hourly):
        """可视化节假日正弦拟合"""
        plt.figure(figsize=(10, 6))

        # 原始数据
        plt.scatter(hours, values, color='blue', alpha=0.7, label='归一化数据', s=50)

        # 拟合曲线
        hours_fine = np.linspace(0, 23, 100)
        fit_values = sin_func(hours_fine, *params)
        plt.plot(hours_fine, fit_values, 'r-', linewidth=2, label=f'正弦拟合 (A={params[0]:.3f}, φ={params[1]:.1f})')

        plt.title('节假日到达率正弦拟合', fontsize=12)
        plt.xlabel('小时')
        plt.ylabel('归一化到达率')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('holiday_sin_fit.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("节假日正弦拟合图表已保存")

    def plot_weekday_factors(self, weekday_stats):
        """可视化星期几调整因子"""
        plt.figure(figsize=(10, 6))

        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        factors = []

        for wd in weekdays:
            if wd in weekday_stats.index:
                factors.append(weekday_stats.loc[wd, 'Adjustment_Factor'])
            else:
                factors.append(1.0)  # 默认值

        colors = plt.cm.Set3(np.linspace(0, 1, len(weekdays)))

        bars = plt.bar(range(len(weekdays)), factors, color=colors, alpha=0.7)
        plt.title('星期几调整因子', fontsize=12)
        plt.xlabel('星期几')
        plt.ylabel('调整因子')
        plt.xticks(range(len(weekdays)), ['周一', '周二', '周三', '周四', '周五'])
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='基准线 (1.0)')

        # 添加数值标签
        for i, v in enumerate(factors):
            plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

        plt.legend()
        plt.tight_layout()
        plt.savefig('weekday_factors.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("星期几调整因子图表已保存")

    def run_estimation(self):
        """运行完整的参数估计流程"""
        print("=" * 60)
        print("开始NHPP模型参数估计")
        print("=" * 60)

        # 1. 估计每小时基础到达率
        weekday_hourly, weekend_hourly = self.estimate_hourly_baseline()

        # 2. 估计交通模式调整因子
        mode_factors = self.estimate_mode_factors()

        # 3. 估计实时调整系数α
        alpha = self.estimate_realtime_alpha()

        # 4. 估计节假日调整系数β
        beta = self.estimate_beta_for_holiday()

        # 5. 估计星期几调整因子
        weekday_factors = self.estimate_weekday_factors()

        # 6. 保存参数
        self.save_parameters()

        print("\n" + "=" * 60)
        print("参数估计完成!")
        print("=" * 60)

        # 返回总结
        return {
            'weekday_hourly': weekday_hourly,
            'weekend_hourly': weekend_hourly,
            'mode_factors': mode_factors,
            'alpha': alpha,
            'beta': beta,
            'weekday_factors': weekday_factors
        }


# 主程序
if __name__ == "__main__":
    try:
        # 创建参数估计器
        estimator = NHPPParameterEstimator()

        # 运行参数估计
        results = estimator.run_estimation()

        print("\n参数估计总结:")
        print(f"- 实时调整系数α: {estimator.params.get('alpha', {}).get('recommended', 'N/A'):.3f}")
        print(f"- 节假日调整系数β: {estimator.params.get('beta', {}).get('recommended', 'N/A'):.3f}")

        # 保存最终的NHPP公式
        with open('nhpp_final_formula.txt', 'w', encoding='utf-8') as f:
            f.write("NHPP最终预测公式\n")
            f.write("=" * 50 + "\n\n")

            f.write("1. 工作日模型:\n")
            f.write(
                "λ_workday(t) = λ_base_workday(h(t)) × f_mode(m(t)) × γ(Weekday) × [1 + α × (R_recent(t) - R_historical(t))/R_historical(t)]\n\n")

            f.write("2. 节假日模型:\n")
            f.write("λ_holiday(t) = λ_base_holiday(h(t)) × [1 + β × sin(2π(h(t)-14)/24)]\n\n")

            f.write("3. 5分钟预测:\n")
            f.write("N̂(t, t+5) = 5 × λ(t)\n\n")

            f.write("4. 95%置信区间:\n")
            f.write("CI = [N̂ - 1.96√N̂, N̂ + 1.96√N̂]\n\n")

            f.write("5. 估计参数:\n")
            alpha_val = estimator.params.get('alpha', {}).get('recommended', 0.3)
            beta_val = estimator.params.get('beta', {}).get('recommended', 0.15)
            f.write(f"   α = {alpha_val:.3f}\n")
            f.write(f"   β = {beta_val:.3f}\n")

        print("NHPP公式已保存到: nhpp_final_formula.txt")

    except Exception as e:
        print(f"参数估计过程中出错: {e}")
        import traceback

        traceback.print_exc()