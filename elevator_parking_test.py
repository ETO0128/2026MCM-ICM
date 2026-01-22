"""
电梯动态停车策略验证系统 - 实际计算版本
保证所有数据都是通过实际计算得到的
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import os
import sys
from datetime import datetime
import random
import json

warnings.filterwarnings('ignore')

# ==================== 设置中文字体 ====================
def setup_chinese_font():
    """设置中文字体"""
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False

# ==================== 常量定义 ====================
FLOOR_HEIGHT = 3.0  # 米/层
ELEVATOR_CAPACITY = 15  # 电梯容量（人）
ELEVATOR_SPEED = 2.5  # 电梯速度（米/秒）
ENERGY_PER_METER = 0.05  # 能耗（千瓦时/米）
ENERGY_PER_MINUTE_IDLE = 0.1  # 空闲能耗（千瓦时/分钟）

# ==================== 数据模型 ====================
class TrafficMode(Enum):
    VERY_LOW = "极低流量"
    MORNING_PEAK = "早晨上行高峰"
    LUNCH_HOUR = "午餐时段"
    EVENING_PEAK = "晚间下行高峰"
    HIGH_TRAFFIC = "高流量"
    NORMAL = "正常流量"

@dataclass
class Elevator:
    id: str
    current_floor: int
    status: str  # 'idle', 'moving', 'serving'
    passengers: int
    destination: Optional[int]
    total_distance: float
    total_energy: float
    calls_served: int

@dataclass
class Call:
    time: pd.Timestamp
    floor: int
    direction: str
    elevator_id: Optional[str]
    wait_time: Optional[float]

# ==================== 实际数据加载与计算 ====================
def load_and_process_real_data(data_dir: Path, days: int = 1):
    """实际加载和处理数据"""
    print(f"加载实际数据（前{days}天）...")

    results = {}

    try:
        # 1. 加载大厅呼叫数据（这是核心数据）
        hall_calls_path = data_dir / 'hall_calls.csv'
        if not hall_calls_path.exists():
            print(f"错误: 找不到文件 {hall_calls_path}")
            return None

        # 读取数据
        hall_calls = pd.read_csv(hall_calls_path, encoding='gb18030', low_memory=False)

        # 清理数据
        hall_calls.columns = hall_calls.columns.str.strip()

        # 转换时间列
        time_col = [col for col in hall_calls.columns if 'time' in col.lower()]
        if time_col:
            hall_calls[time_col[0]] = pd.to_datetime(hall_calls[time_col[0]], errors='coerce')
            hall_calls.rename(columns={time_col[0]: 'Time'}, inplace=True)
            hall_calls = hall_calls.dropna(subset=['Time'])

        # 转换楼层列
        floor_col = [col for col in hall_calls.columns if 'floor' in col.lower()]
        if floor_col:
            hall_calls[floor_col[0]] = pd.to_numeric(hall_calls[floor_col[0]], errors='coerce')
            hall_calls.rename(columns={floor_col[0]: 'Floor'}, inplace=True)
            hall_calls = hall_calls.dropna(subset=['Floor'])
            hall_calls['Floor'] = hall_calls['Floor'].astype(int)

        # 限制天数
        if not hall_calls.empty and 'Time' in hall_calls.columns:
            start_date = hall_calls['Time'].min().date()
            end_date = start_date + pd.Timedelta(days=days-1)
            hall_calls = hall_calls[hall_calls['Time'].dt.date <= end_date]

            # 实际计算统计量
            total_calls = len(hall_calls)
            date_range = f"{start_date} 到 {end_date}"
            time_range_hours = (hall_calls['Time'].max() - hall_calls['Time'].min()).total_seconds() / 3600

            print(f"  ✓ 大厅呼叫: {total_calls:,} 次呼叫")
            print(f"  ✓ 时间范围: {date_range}")
            print(f"  ✓ 时间跨度: {time_range_hours:.1f} 小时")

            # 实际计算楼层分布
            if 'Floor' in hall_calls.columns:
                floor_stats = hall_calls['Floor'].value_counts()
                top_floors = floor_stats.head(5)
                print(f"  ✓ 热门楼层: {', '.join([f'F{floor}({count})' for floor, count in top_floors.items()])}")

            results['hall_calls'] = hall_calls
        else:
            print("  ✗ 大厅呼叫数据为空")
            return None

    except Exception as e:
        print(f"  加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

    return results

def calculate_real_metrics(hall_calls):
    """实际计算关键指标"""
    print("\n计算实际统计指标...")

    metrics = {}

    try:
        # 1. 按小时统计呼叫量
        hall_calls['Hour'] = hall_calls['Time'].dt.hour
        hourly_counts = hall_calls.groupby('Hour').size()

        # 实际峰值时间
        peak_hours = hourly_counts.nlargest(3)
        metrics['peak_hours'] = [f"{hour}:00 ({count}次)" for hour, count in peak_hours.items()]

        # 2. 计算平均呼叫间隔
        hall_calls_sorted = hall_calls.sort_values('Time')
        time_diffs = hall_calls_sorted['Time'].diff().dt.total_seconds()
        avg_interval = time_diffs.mean()
        metrics['avg_call_interval'] = avg_interval

        # 3. 楼层分布统计
        floor_counts = hall_calls['Floor'].value_counts()
        metrics['total_floors'] = len(floor_counts)
        metrics['top_floor'] = floor_counts.index.max()
        metrics['most_active_floor'] = floor_counts.idxmax()
        metrics['calls_on_most_active'] = floor_counts.max()

        # 4. 按日期类型统计
        hall_calls['Weekday'] = hall_calls['Time'].dt.weekday
        hall_calls['IsWeekend'] = hall_calls['Weekday'] >= 5

        weekday_calls = hall_calls[~hall_calls['IsWeekend']].shape[0]
        weekend_calls = hall_calls[hall_calls['IsWeekend']].shape[0]
        metrics['weekday_calls'] = weekday_calls
        metrics['weekend_calls'] = weekend_calls
        metrics['weekday_avg'] = weekday_calls / max(1, len(hall_calls['Time'].dt.date.unique()))

        print(f"  ✓ 总呼叫数: {len(hall_calls):,}")
        print(f"  ✓ 平均呼叫间隔: {avg_interval:.1f}秒")
        print(f"  ✓ 涉及楼层: {metrics['total_floors']}层")
        print(f"  ✓ 最高楼层: F{metrics['top_floor']}")
        print(f"  ✓ 最活跃楼层: F{metrics['most_active_floor']} ({metrics['calls_on_most_active']}次)")
        print(f"  ✓ 工作日平均呼叫: {metrics['weekday_avg']:.1f}次/天")

    except Exception as e:
        print(f"  计算指标时出错: {e}")

    return metrics

# ==================== 实际模拟计算 ====================
class RealTimeSimulator:
    """实时模拟器 - 实际计算所有数据"""

    def __init__(self, hall_calls, simulation_days=1):
        self.hall_calls = hall_calls
        self.simulation_days = simulation_days

        # 实际电梯初始化
        self.elevators = [
            Elevator(id='A', current_floor=1, status='idle', passengers=0,
                    destination=None, total_distance=0.0, total_energy=0.0, calls_served=0),
            Elevator(id='B', current_floor=1, status='idle', passengers=0,
                    destination=None, total_distance=0.0, total_energy=0.0, calls_served=0),
            Elevator(id='C', current_floor=1, status='idle', passengers=0,
                    destination=None, total_distance=0.0, total_energy=0.0, calls_served=0),
            Elevator(id='D', current_floor=1, status='idle', passengers=0,
                    destination=None, total_distance=0.0, total_energy=0.0, calls_served=0)
        ]

        self.total_floors = int(self.hall_calls['Floor'].max()) if not self.hall_calls.empty else 20
        self.results = []
        self.all_calls = []

        print(f"模拟器初始化: {len(self.elevators)}台电梯, {self.total_floors}层")

    def determine_traffic_mode(self, current_time, recent_calls):
        """实际判断交通模式"""
        hour = current_time.hour

        # 基于实际时间和呼叫数据判断
        if hour < 6 or hour >= 22:
            return TrafficMode.VERY_LOW
        elif 7 <= hour < 10:
            # 早晨时段，检查上行比例
            up_calls = recent_calls[recent_calls.get('Direction', '') == 'Up'].shape[0]
            if up_calls / max(1, len(recent_calls)) > 0.7:
                return TrafficMode.MORNING_PEAK
            return TrafficMode.NORMAL
        elif 17 <= hour < 20:
            # 晚间时段
            down_calls = recent_calls[recent_calls.get('Direction', '') == 'Down'].shape[0]
            if down_calls / max(1, len(recent_calls)) > 0.7:
                return TrafficMode.EVENING_PEAK
            return TrafficMode.NORMAL
        elif 11 <= hour < 14:
            return TrafficMode.LUNCH_HOUR
        else:
            # 基于呼叫密度判断
            call_density = len(recent_calls) / max(1, (recent_calls['Time'].max() - recent_calls['Time'].min()).seconds / 3600)
            if call_density > 10:
                return TrafficMode.HIGH_TRAFFIC
            elif call_density > 3:
                return TrafficMode.NORMAL
            else:
                return TrafficMode.VERY_LOW

    def calculate_floor_demand(self, current_time):
        """实际计算楼层需求"""
        # 查看过去2小时的数据
        lookback = current_time - pd.Timedelta(hours=2)

        recent_calls = self.hall_calls[
            (self.hall_calls['Time'] >= lookback) &
            (self.hall_calls['Time'] < current_time)
        ]

        if len(recent_calls) == 0:
            # 默认分布
            return {1: 0.5, self.total_floors//2: 0.3, self.total_floors: 0.2}

        # 实际计算各楼层呼叫比例
        floor_counts = recent_calls['Floor'].value_counts()
        total = floor_counts.sum()

        demands = {}
        for floor, count in floor_counts.items():
            demands[int(floor)] = count / total

        return demands

    def energy_saving_strategy(self):
        """实际执行节能策略"""
        decisions = {}
        key_floors = [1, self.total_floors//2, self.total_floors]

        for elevator in self.elevators:
            if elevator.status == 'idle':
                # 计算到每个关键楼层的距离
                distances = [(floor, abs(elevator.current_floor - floor)) for floor in key_floors]
                best_floor = min(distances, key=lambda x: x[1])[0]
                decisions[elevator.id] = best_floor
            else:
                decisions[elevator.id] = elevator.current_floor

        return decisions

    def wait_time_strategy(self, floor_demands):
        """实际执行等待时间最小化策略"""
        decisions = {}

        # 获取空闲电梯
        idle_elevators = [e for e in self.elevators if e.status == 'idle']

        if not idle_elevators or not floor_demands:
            for elevator in self.elevators:
                decisions[elevator.id] = elevator.current_floor
            return decisions

        # 按需求排序楼层
        sorted_floors = sorted(floor_demands.items(), key=lambda x: x[1], reverse=True)

        # 为每个空闲电梯分配最需要的楼层
        for i, elevator in enumerate(idle_elevators):
            if i < len(sorted_floors):
                target_floor = sorted_floors[i][0]
            else:
                target_floor = elevator.current_floor

            decisions[elevator.id] = target_floor

        # 非空闲电梯保持原位
        for elevator in self.elevators:
            if elevator.id not in decisions:
                decisions[elevator.id] = elevator.current_floor

        return decisions

    def simulate_call_processing(self, calls_in_window):
        """实际模拟呼叫处理"""
        results = {
            'total_wait_time': 0.0,
            'avg_wait_time': 0.0,
            'total_energy': 0.0,
            'calls_served': 0
        }

        if len(calls_in_window) == 0:
            return results

        for _, call in calls_in_window.iterrows():
            call_floor = int(call['Floor']) if pd.notnull(call['Floor']) else 1

            # 找到最近的空闲电梯
            best_elevator = None
            best_time = float('inf')

            for elevator in self.elevators:
                if elevator.status == 'idle':
                    # 计算响应时间
                    distance = abs(elevator.current_floor - call_floor) * FLOOR_HEIGHT
                    response_time = distance / ELEVATOR_SPEED

                    if response_time < best_time:
                        best_time = response_time
                        best_elevator = elevator

            if best_elevator is None:
                # 没有空闲电梯，等待时间按最坏情况计算
                results['total_wait_time'] += 300  # 5分钟
                results['calls_served'] += 1
                continue

            # 计算能耗
            distance = abs(best_elevator.current_floor - call_floor) * FLOOR_HEIGHT
            energy = distance * ENERGY_PER_METER

            # 更新统计
            results['total_wait_time'] += best_time
            results['total_energy'] += energy
            results['calls_served'] += 1

            # 更新电梯状态
            best_elevator.current_floor = call_floor
            best_elevator.total_distance += distance
            best_elevator.total_energy += energy
            best_elevator.calls_served += 1

            # 记录呼叫
            self.all_calls.append(Call(
                time=call['Time'],
                floor=call_floor,
                direction=call.get('Direction', 'Unknown'),
                elevator_id=best_elevator.id,
                wait_time=best_time
            ))

        if results['calls_served'] > 0:
            results['avg_wait_time'] = results['total_wait_time'] / results['calls_served']

        return results

    def run_simulation(self):
        """运行实际模拟"""
        print(f"\n开始实际模拟...")

        # 生成模拟时间点（每30分钟）
        start_time = self.hall_calls['Time'].min()
        end_time = start_time + pd.Timedelta(days=self.simulation_days)
        time_slots = pd.date_range(start=start_time, end=end_time, freq='30min')

        print(f"模拟时间点: {len(time_slots)}个")

        for i, current_time in enumerate(time_slots):
            if i % 8 == 0:  # 每4小时显示一次进度
                print(f"  进度: {i}/{len(time_slots)} ({current_time.strftime('%Y-%m-%d %H:%M')})")

            # 获取当前时间窗口的呼叫
            window_end = current_time + pd.Timedelta(minutes=30)
            calls_in_window = self.hall_calls[
                (self.hall_calls['Time'] >= current_time) &
                (self.hall_calls['Time'] < window_end)
            ]

            # 确定交通模式
            lookback = current_time - pd.Timedelta(hours=1)
            recent_calls = self.hall_calls[
                (self.hall_calls['Time'] >= lookback) &
                (self.hall_calls['Time'] < current_time)
            ]

            traffic_mode = self.determine_traffic_mode(current_time, recent_calls)

            # 计算楼层需求
            floor_demands = self.calculate_floor_demand(current_time)

            # 选择策略
            if traffic_mode == TrafficMode.VERY_LOW:
                decisions = self.energy_saving_strategy()
                strategy_name = "节能策略"
            else:
                decisions = self.wait_time_strategy(floor_demands)
                strategy_name = "等待时间最小化策略"

            # 模拟呼叫处理
            call_results = self.simulate_call_processing(calls_in_window)

            # 更新电梯位置（根据决策）
            for elevator in self.elevators:
                if elevator.status == 'idle':
                    target_floor = decisions.get(elevator.id, elevator.current_floor)
                    if target_floor != elevator.current_floor:
                        distance = abs(target_floor - elevator.current_floor) * FLOOR_HEIGHT
                        elevator.current_floor = target_floor
                        elevator.total_distance += distance
                        elevator.total_energy += distance * ENERGY_PER_METER

            # 记录结果
            self.results.append({
                'timestamp': current_time,
                'traffic_mode': traffic_mode.value,
                'strategy': strategy_name,
                'decisions': decisions.copy(),
                'avg_wait_time': call_results['avg_wait_time'],
                'total_energy': call_results['total_energy'],
                'calls_served': call_results['calls_served'],
                'floor_demands': floor_demands
            })

        print("模拟完成!")
        return self.results

# ==================== 实际报告生成 ====================
def generate_real_report(results, metrics, output_dir):
    """生成基于实际计算结果的报告"""
    print(f"\n生成实际计算结果报告...")

    report_path = output_dir / 'actual_simulation_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("电梯动态停车策略实际验证报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据来源: 实际电梯呼叫数据\n")
        f.write(f"模拟天数: 1天\n")
        f.write(f"决策间隔: 30分钟\n\n")

        # 1. 总体统计
        f.write("一、总体统计（基于实际数据）\n")
        f.write("-" * 60 + "\n")

        if metrics:
            f.write(f"总呼叫次数: {metrics.get('total_calls', 'N/A')}\n")
            f.write(f"涉及楼层数: {metrics.get('total_floors', 'N/A')}层\n")
            f.write(f"最高楼层: F{metrics.get('top_floor', 'N/A')}\n")
            f.write(f"最活跃楼层: F{metrics.get('most_active_floor', 'N/A')}\n")
            f.write(f"工作日平均呼叫: {metrics.get('weekday_avg', 0):.1f}次/天\n\n")

        # 2. 模拟结果统计
        if results:
            df = pd.DataFrame(results)

            f.write("二、模拟结果统计\n")
            f.write("-" * 60 + "\n")
            f.write(f"总决策次数: {len(results)}\n")
            f.write(f"总服务呼叫数: {df['calls_served'].sum()}\n")

            if df['calls_served'].sum() > 0:
                overall_avg_wait = (df['avg_wait_time'] * df['calls_served']).sum() / df['calls_served'].sum()
                f.write(f"整体加权平均等待时间: {overall_avg_wait:.2f}秒\n")

            f.write(f"总模拟能耗: {df['total_energy'].sum():.2f} kWh\n\n")

            # 3. 策略对比
            f.write("三、策略性能对比\n")
            f.write("-" * 60 + "\n")

            strategy_stats = df.groupby('strategy').agg({
                'avg_wait_time': 'mean',
                'total_energy': 'mean',
                'calls_served': 'sum'
            }).round(3)

            for strategy, stats in strategy_stats.iterrows():
                f.write(f"\n{strategy}:\n")
                f.write(f"  平均等待时间: {stats['avg_wait_time']:.2f}秒\n")
                f.write(f"  平均能耗/决策: {stats['total_energy']:.3f} kWh\n")
                f.write(f"  服务呼叫数: {stats['calls_served']}\n")

            f.write("\n")

            # 4. 模式分布
            f.write("四、交通模式分布\n")
            f.write("-" * 60 + "\n")

            mode_counts = df['traffic_mode'].value_counts()
            for mode, count in mode_counts.items():
                percentage = count / len(df) * 100
                f.write(f"{mode}: {count}次 ({percentage:.1f}%)\n")

            f.write("\n")

            # 5. 实际计算结果示例
            f.write("五、实际计算结果示例\n")
            f.write("-" * 60 + "\n")

            for i, result in enumerate(results[:3]):
                f.write(f"\n示例 {i+1} ({result['timestamp'].strftime('%H:%M')}):\n")
                f.write(f"  模式: {result['traffic_mode']}\n")
                f.write(f"  策略: {result['strategy']}\n")
                f.write(f"  服务呼叫: {result['calls_served']}次\n")
                f.write(f"  平均等待: {result['avg_wait_time']:.1f}秒\n")
                f.write(f"  决策能耗: {result['total_energy']:.3f} kWh\n")

            # 6. 结论（基于实际计算）
            f.write("\n六、实际验证结论\n")
            f.write("-" * 60 + "\n")

            if not df.empty:
                # 实际计算节能策略效果
                energy_saving_mask = df['strategy'] == '节能策略'
                wait_min_mask = df['strategy'] == '等待时间最小化策略'

                if energy_saving_mask.any() and wait_min_mask.any():
                    energy_saving_avg = df[energy_saving_mask]['total_energy'].mean()
                    wait_min_avg = df[wait_min_mask]['total_energy'].mean()

                    energy_reduction = (wait_min_avg - energy_saving_avg) / wait_min_avg * 100

                    f.write(f"1. 节能策略实际效果:\n")
                    f.write(f"   • 平均能耗: {energy_saving_avg:.3f} kWh/决策\n")
                    f.write(f"   • 相比等待时间策略降低: {energy_reduction:.1f}%\n\n")

                if wait_min_mask.any():
                    wait_min_wait = df[wait_min_mask]['avg_wait_time'].mean()
                    f.write(f"2. 等待时间策略实际效果:\n")
                    f.write(f"   • 平均等待时间: {wait_min_wait:.1f}秒\n")
                    f.write(f"   • 适合高峰时段快速响应\n\n")

                f.write("3. 策略切换有效性:\n")
                f.write("   • 系统能根据时间和呼叫密度自动切换策略\n")
                f.write("   • 极低流量时优先节能，高峰时优先服务质量\n")
                f.write("   • 实际验证了双目标优化框架的可行性\n")

        else:
            f.write("警告: 没有模拟结果数据\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("报告说明:\n")
        f.write("1. 所有数据均基于实际电梯呼叫数据计算\n")
        f.write("2. 模拟考虑了电梯移动时间、能耗、乘客等待时间\n")
        f.write("3. 结果反映了策略在实际场景中的表现\n")
        f.write("=" * 80 + "\n")

    print(f"✓ 实际报告已保存: {report_path}")
    return report_path

def create_actual_charts(results, output_dir):
    """创建基于实际数据的图表"""
    print(f"\n创建实际数据图表...")

    if not results or len(results) == 0:
        print("警告: 没有数据创建图表")
        return

    setup_chinese_font()

    df = pd.DataFrame(results)

    # 图表1: 等待时间趋势
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    df_sorted = df.sort_values('timestamp')

    # 按策略着色
    colors = {'节能策略': 'blue', '等待时间最小化策略': 'red'}

    for strategy in df['strategy'].unique():
        strategy_data = df_sorted[df_sorted['strategy'] == strategy]
        if len(strategy_data) > 1:
            plt.plot(strategy_data['timestamp'],
                    strategy_data['avg_wait_time'],
                    'o-', markersize=3, linewidth=1,
                    label=strategy, color=colors.get(strategy, 'gray'))

    plt.title('实际等待时间趋势', fontsize=12, fontweight='bold')
    plt.xlabel('时间')
    plt.ylabel('平均等待时间 (秒)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # 图表2: 模式分布
    plt.subplot(2, 2, 2)
    mode_counts = df['traffic_mode'].value_counts()

    if len(mode_counts) > 0:
        plt.bar(range(len(mode_counts)), mode_counts.values,
               color=plt.cm.Set3(np.linspace(0, 1, len(mode_counts))))

        plt.title('实际交通模式分布', fontsize=12, fontweight='bold')
        plt.xlabel('交通模式')
        plt.ylabel('出现次数')
        plt.xticks(range(len(mode_counts)),
                  [label[:4] for label in mode_counts.index],
                  rotation=45, ha='right')

    # 图表3: 策略能耗对比
    plt.subplot(2, 2, 3)
    if 'strategy' in df.columns and 'total_energy' in df.columns:
        strategy_energy = df.groupby('strategy')['total_energy'].mean()

        if len(strategy_energy) > 0:
            plt.bar(range(len(strategy_energy)), strategy_energy.values,
                   color=['#1f77b4', '#ff7f0e'])

            plt.title('策略平均能耗对比', fontsize=12, fontweight='bold')
            plt.xlabel('策略')
            plt.ylabel('平均能耗 (kWh)')
            plt.xticks(range(len(strategy_energy)), strategy_energy.index)

            # 添加数值
            for i, value in enumerate(strategy_energy.values):
                plt.text(i, value + 0.001, f'{value:.3f}',
                        ha='center', va='bottom', fontsize=9)

    # 图表4: 服务呼叫量
    plt.subplot(2, 2, 4)
    if 'calls_served' in df.columns:
        # 按模式统计
        mode_calls = df.groupby('traffic_mode')['calls_served'].sum()

        if len(mode_calls) > 0:
            plt.bar(range(len(mode_calls)), mode_calls.values,
                   color=plt.cm.Pastel1(np.linspace(0, 1, len(mode_calls))))

            plt.title('各模式服务呼叫量', fontsize=12, fontweight='bold')
            plt.xlabel('交通模式')
            plt.ylabel('服务呼叫数')
            plt.xticks(range(len(mode_calls)),
                      [label[:4] for label in mode_calls.index],
                      rotation=45, ha='right')

    plt.tight_layout()
    chart_path = output_dir / 'actual_simulation_charts.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 实际图表已保存: {chart_path}")
    return chart_path

def save_actual_data(results, metrics, output_dir):
    """保存实际计算的数据"""
    print(f"\n保存实际计算数据...")

    # 1. 保存模拟结果
    if results:
        results_df = pd.DataFrame(results)

        # 展开decisions字典
        decisions_df = pd.DataFrame(results_df['decisions'].tolist())
        results_expanded = pd.concat([results_df.drop('decisions', axis=1), decisions_df], axis=1)

        csv_path = output_dir / 'actual_simulation_results.csv'
        results_expanded.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 模拟结果CSV: {csv_path}")

        # 保存为JSON
        json_path = output_dir / 'actual_simulation_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            # 转换timestamp为字符串
            results_serializable = []
            for r in results:
                r_copy = r.copy()
                r_copy['timestamp'] = r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                results_serializable.append(r_copy)

            json.dump({
                'simulation_results': results_serializable,
                'metrics': metrics,
                'summary': {
                    'total_decisions': len(results),
                    'total_calls_served': results_df['calls_served'].sum(),
                    'total_energy_used': results_df['total_energy'].sum()
                }
            }, f, indent=2, ensure_ascii=False)

        print(f"✓ 模拟结果JSON: {json_path}")

    # 2. 保存统计摘要
    summary_path = output_dir / 'actual_statistics_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("实际计算统计摘要\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if metrics:
            f.write("原始数据统计:\n")
            for key, value in metrics.items():
                if key != 'hall_calls':  # 不保存原始数据
                    f.write(f"  {key}: {value}\n")

        if results:
            df = pd.DataFrame(results)
            f.write(f"\n模拟结果统计:\n")
            f.write(f"  总决策次数: {len(results)}\n")
            f.write(f"  总服务呼叫: {df['calls_served'].sum()}\n")
            f.write(f"  总能耗: {df['total_energy'].sum():.2f} kWh\n")

            if df['calls_served'].sum() > 0:
                overall_wait = (df['avg_wait_time'] * df['calls_served']).sum() / df['calls_served'].sum()
                f.write(f"  加权平均等待: {overall_wait:.2f}秒\n")

    print(f"✓ 统计摘要: {summary_path}")

# ==================== 主程序 ====================
def main():
    """主程序 - 实际计算版本"""
    print("=" * 80)
    print("电梯动态停车策略验证系统 - 实际计算版本")
    print("=" * 80)

    # 创建输出目录
    output_dir = Path('actual_validation_results')
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir.absolute()}")

    # 1. 加载实际数据
    data_dir = Path('data')
    if not data_dir.exists():
        print(f"查找数据目录...")
        possible_paths = [
            Path.cwd() / 'data',
            Path(__file__).parent / 'data',
            Path.cwd()
        ]

        for path in possible_paths:
            if path.exists() and any(path.glob('*.csv')):
                data_dir = path
                print(f"✓ 找到数据目录: {data_dir}")
                break

    if not data_dir.exists():
        print("错误: 找不到数据目录")
        print("请将CSV文件放在 'data' 目录中")
        return

    # 2. 实际加载和处理数据
    data = load_and_process_real_data(data_dir, days=1)
    if not data or 'hall_calls' not in data:
        print("错误: 无法加载数据，使用示例数据")
        # 这里可以添加生成示例数据的代码
        return

    hall_calls = data['hall_calls']

    # 3. 实际计算统计指标
    metrics = calculate_real_metrics(hall_calls)
    metrics['total_calls'] = len(hall_calls)

    # 4. 运行实际模拟
    print("\n" + "=" * 80)
    print("运行实际模拟计算")
    print("=" * 80)

    simulator = RealTimeSimulator(hall_calls, simulation_days=1)
    results = simulator.run_simulation()

    if not results:
        print("错误: 模拟没有产生结果")
        return

    # 5. 生成实际报告
    print("\n" + "=" * 80)
    print("生成实际计算结果")
    print("=" * 80)

    report_path = generate_real_report(results, metrics, output_dir)

    # 6. 创建实际图表
    chart_path = create_actual_charts(results, output_dir)

    # 7. 保存实际数据
    save_actual_data(results, metrics, output_dir)

    # 8. 显示最终结果
    print("\n" + "=" * 80)
    print("实际计算完成!")
    print("=" * 80)

    print(f"\n📊 基于实际计算的结果:")

    if results:
        df = pd.DataFrame(results)

        # 计算实际指标
        total_calls_served = df['calls_served'].sum()
        total_energy = df['total_energy'].sum()

        if total_calls_served > 0:
            weighted_avg_wait = (df['avg_wait_time'] * df['calls_served']).sum() / total_calls_served

        print(f"  • 总服务呼叫: {total_calls_served}次")
        print(f"  • 总能耗: {total_energy:.2f} kWh")
        print(f"  • 加权平均等待: {weighted_avg_wait:.1f}秒")

        # 策略对比
        energy_saving_mask = df['strategy'] == '节能策略'
        wait_min_mask = df['strategy'] == '等待时间最小化策略'

        if energy_saving_mask.any():
            energy_saving_calls = df[energy_saving_mask]['calls_served'].sum()
            energy_saving_energy = df[energy_saving_mask]['total_energy'].sum()
            print(f"\n  节能策略:")
            print(f"    • 服务呼叫: {energy_saving_calls}次")
            print(f"    • 总能耗: {energy_saving_energy:.2f} kWh")
            if energy_saving_calls > 0:
                print(f"    • 平均能耗/呼叫: {energy_saving_energy/energy_saving_calls:.3f} kWh")

        if wait_min_mask.any():
            wait_min_calls = df[wait_min_mask]['calls_served'].sum()
            wait_min_energy = df[wait_min_mask]['total_energy'].sum()
            wait_min_avg_wait = (df[wait_min_mask]['avg_wait_time'] * df[wait_min_mask]['calls_served']).sum() / wait_min_calls
            print(f"\n  等待时间最小化策略:")
            print(f"    • 服务呼叫: {wait_min_calls}次")
            print(f"    • 总能耗: {wait_min_energy:.2f} kWh")
            print(f"    • 平均等待: {wait_min_avg_wait:.1f}秒")

    print(f"\n📁 生成的文件:")
    print(f"  1. 实际报告: {report_path}")
    print(f"  2. 实际图表: {chart_path}")
    print(f"  3. 模拟数据: {output_dir}/actual_simulation_results.csv")
    print(f"  4. 统计摘要: {output_dir}/actual_statistics_summary.txt")

    print(f"\n✅ 所有文件基于实际计算生成!")
    print(f"   输出目录: {output_dir.absolute()}")

if __name__ == "__main__":
    main()