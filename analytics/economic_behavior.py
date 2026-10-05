"""
游戏内玩家经济行为分析与消费预测系统 - 核心模块
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import time

SECONDS_PER_DAY = 86400.0


@dataclass
class Config:
    whale_threshold: float = 1000.0
    dolphin_threshold: float = 200.0
    minnow_threshold: float = 1.0
    churn_warning_threshold: float = 0.7
    ltv_horizon_months: int = 12


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


class EconomicBehavior:
    """玩家经济行为分析、消费预测、付费意愿、玩家价值与流失预测。"""

    FEATURE_NAMES = (
        "transaction_count",
        "total_spent",
        "avg_amount",
        "purchase_recency",
        "purchase_frequency",
        "active_days",
    )
    _DEFAULT_WEIGHTS = [0.8, 1.0, 0.5, 1.2, 0.8, 0.5]
    _DEFAULT_BIAS = -1.5

    def __init__(self, config: Optional[Dict] = None):
        self.config = Config(**(config or {}))
        self._state = {}
        self._history = []
        self._players = {}
        self._weights = list(self._DEFAULT_WEIGHTS)
        self._bias = self._DEFAULT_BIAS
        self._trained = False

    # ------------------------------------------------------------------
    # 数据记录
    # ------------------------------------------------------------------
    def _player(self, player_id: str) -> Dict:
        if player_id not in self._players:
            self._players[player_id] = {
                "transactions": [],
                "logins": [],
                "first_seen": None,
            }
        return self._players[player_id]

    def record_transaction(self, player_id: str, amount: float,
                           timestamp: Optional[float] = None,
                           category: str = "general") -> Dict:
        ts = time.time() if timestamp is None else float(timestamp)
        tx = {
            "player_id": player_id,
            "amount": float(amount),
            "timestamp": ts,
            "category": category,
        }
        player = self._player(player_id)
        player["transactions"].append(tx)
        if player["first_seen"] is None or ts < player["first_seen"]:
            player["first_seen"] = ts
        self._history.append(tx)
        return tx

    def record_login(self, player_id: str, timestamp: Optional[float] = None) -> float:
        ts = time.time() if timestamp is None else float(timestamp)
        player = self._player(player_id)
        player["logins"].append(ts)
        if player["first_seen"] is None or ts < player["first_seen"]:
            player["first_seen"] = ts
        return ts

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self._players = {}
        self._weights = list(self._DEFAULT_WEIGHTS)
        self._bias = self._DEFAULT_BIAS
        self._trained = False

    # ------------------------------------------------------------------
    # 经济行为分析：消费模式 + 消费习惯
    # ------------------------------------------------------------------
    def behavior_analysis(self, player_id: str, now: Optional[float] = None) -> Dict:
        player = self._players.get(player_id)
        txs = list(player["transactions"]) if player else []
        amounts = [t["amount"] for t in txs]
        total = sum(amounts)
        count = len(amounts)
        return {
            "player_id": player_id,
            "total_spent": total,
            "transaction_count": count,
            "avg_amount": total / count if count else 0.0,
            "max_amount": max(amounts) if amounts else 0.0,
            "min_amount": min(amounts) if amounts else 0.0,
            "consumption_pattern": self._classify_pattern(txs),
            "consumption_habits": self._analyze_habits(txs),
        }

    def _classify_pattern(self, txs: List[Dict]) -> str:
        if not txs:
            return "non_payer"
        if len(txs) == 1:
            return "one_time"
        total = sum(t["amount"] for t in txs)
        if total >= self.config.whale_threshold:
            return "high_roller"
        timestamps = sorted(t["timestamp"] for t in txs)
        intervals = [b - a for a, b in zip(timestamps, timestamps[1:])]
        if intervals:
            mean_iv = sum(intervals) / len(intervals)
            if mean_iv > 0:
                variance = sum((iv - mean_iv) ** 2 for iv in intervals) / len(intervals)
                if math.sqrt(variance) / mean_iv < 0.5 and len(txs) >= 3:
                    return "regular_spender"
        return "occasional_spender"

    def _analyze_habits(self, txs: List[Dict]) -> Dict:
        if not txs:
            return {
                "favorite_category": None,
                "preferred_hour": None,
                "avg_interval_days": 0.0,
                "purchases_per_week": 0.0,
                "spending_trend": "none",
            }
        categories = {}
        for t in txs:
            categories[t["category"]] = categories.get(t["category"], 0) + 1
        favorite_category = max(categories.items(), key=lambda kv: kv[1])[0]
        hours = {}
        for t in txs:
            hour = time.gmtime(t["timestamp"]).tm_hour
            hours[hour] = hours.get(hour, 0) + 1
        preferred_hour = max(hours.items(), key=lambda kv: kv[1])[0]
        timestamps = sorted(t["timestamp"] for t in txs)
        intervals = [b - a for a, b in zip(timestamps, timestamps[1:])]
        avg_interval_days = (sum(intervals) / len(intervals) / SECONDS_PER_DAY) if intervals else 0.0
        span_days = max((timestamps[-1] - timestamps[0]) / SECONDS_PER_DAY, 1.0)
        purchases_per_week = len(txs) / span_days * 7.0
        half = len(txs) // 2
        if half >= 1 and len(txs) >= 2:
            first_avg = sum(t["amount"] for t in txs[:half]) / half
            second_avg = sum(t["amount"] for t in txs[half:]) / (len(txs) - half)
            if first_avg > 0 and second_avg > first_avg * 1.2:
                trend = "increasing"
            elif second_avg < first_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        return {
            "favorite_category": favorite_category,
            "preferred_hour": preferred_hour,
            "avg_interval_days": avg_interval_days,
            "purchases_per_week": purchases_per_week,
            "spending_trend": trend,
        }

    # ------------------------------------------------------------------
    # 消费预测：特征 + 逻辑回归模型
    # ------------------------------------------------------------------
    def extract_features(self, player_id: str, now: Optional[float] = None) -> Dict:
        now = time.time() if now is None else float(now)
        player = self._players.get(player_id)
        txs = player["transactions"] if player else []
        logins = player["logins"] if player else []
        count = len(txs)
        total = sum(t["amount"] for t in txs)
        if txs:
            last_purchase = max(t["timestamp"] for t in txs)
            days_since = max(0.0, (now - last_purchase) / SECONDS_PER_DAY)
            first = min(t["timestamp"] for t in txs)
            span_weeks = max((last_purchase - first) / SECONDS_PER_DAY / 7.0, 1.0 / 7.0)
            frequency = count / span_weeks
        else:
            days_since = 365.0
            frequency = 0.0
        first_seen = player["first_seen"] if player and player["first_seen"] is not None else now
        active_days = len({int(ts // SECONDS_PER_DAY) for ts in logins})
        return {
            "transaction_count": count,
            "total_spent": total,
            "avg_amount": total / count if count else 0.0,
            "days_since_last_purchase": days_since,
            "purchase_frequency": frequency,
            "active_days": active_days,
            "account_age_days": max(0.0, (now - first_seen) / SECONDS_PER_DAY),
        }

    def _feature_vector(self, features: Dict) -> List[float]:
        recency = max(0.0, 30.0 - features["days_since_last_purchase"]) / 30.0
        return [
            features["transaction_count"] / 10.0,
            features["total_spent"] / 100.0,
            features["avg_amount"] / 50.0,
            recency,
            features["purchase_frequency"] / 2.0,
            features["active_days"] / 30.0,
        ]

    def train(self, samples: List[Dict], labels: List[int],
              lr: float = 0.5, epochs: int = 500) -> Dict:
        """用标注样本（extract_features 的特征字典 + 0/1 标签）训练逻辑回归模型。"""
        if not samples or len(samples) != len(labels):
            raise ValueError("samples 与 labels 长度必须一致且非空")
        xs = [self._feature_vector(s) for s in samples]
        n = len(xs)
        dim = len(self._weights)
        for _ in range(epochs):
            grad_w = [0.0] * dim
            grad_b = 0.0
            for x, y in zip(xs, labels):
                p = _sigmoid(sum(w * v for w, v in zip(self._weights, x)) + self._bias)
                err = p - y
                for i in range(dim):
                    grad_w[i] += err * x[i]
                grad_b += err
            for i in range(dim):
                self._weights[i] -= lr * grad_w[i] / n
            self._bias -= lr * grad_b / n
        self._trained = True
        return {"trained": True, "samples": n,
                "weights": list(self._weights), "bias": self._bias}

    def consumption_prediction(self, player_id: str, now: Optional[float] = None) -> Dict:
        features = self.extract_features(player_id, now)
        x = self._feature_vector(features)
        probability = _sigmoid(sum(w * v for w, v in zip(self._weights, x)) + self._bias)
        return {
            "player_id": player_id,
            "will_spend": probability >= 0.5,
            "probability": probability,
            "features": features,
            "model": {
                "type": "logistic_regression",
                "trained": self._trained,
                "feature_names": list(self.FEATURE_NAMES),
            },
        }

    # ------------------------------------------------------------------
    # 付费意愿：评估 + 分层
    # ------------------------------------------------------------------
    def willingness_to_pay(self, player_id: str, now: Optional[float] = None) -> Dict:
        now = time.time() if now is None else float(now)
        analysis = self.behavior_analysis(player_id, now)
        total = analysis["total_spent"]
        count = analysis["transaction_count"]
        habits = analysis["consumption_habits"]
        spend_score = min(40.0, total / self.config.whale_threshold * 40.0)
        freq_score = min(20.0, count * 2.0)
        player = self._players.get(player_id)
        txs = player["transactions"] if player else []
        if txs:
            days_since = (now - max(t["timestamp"] for t in txs)) / SECONDS_PER_DAY
            recency_score = max(0.0, 20.0 * (1.0 - days_since / 30.0))
        else:
            recency_score = 0.0
        trend = habits["spending_trend"]
        trend_score = {"increasing": 20.0, "stable": 10.0,
                       "decreasing": 5.0}.get(trend, 0.0)
        score = round(min(100.0, spend_score + freq_score + recency_score + trend_score), 2)
        if score >= 75.0:
            tier = "whale"
        elif score >= 50.0:
            tier = "dolphin"
        elif score >= 25.0:
            tier = "minnow"
        else:
            tier = "free"
        max_amount = analysis["max_amount"]
        estimated_wtp = max_amount * (1.0 + score / 100.0) if txs else 0.0
        return {
            "player_id": player_id,
            "wtp_score": score,
            "estimated_willingness_to_pay": estimated_wtp,
            "tier": tier,
            "evaluation": {
                "spend_score": spend_score,
                "frequency_score": freq_score,
                "recency_score": recency_score,
                "trend_score": trend_score,
            },
        }

    # ------------------------------------------------------------------
    # 玩家价值：LTV + ARPU
    # ------------------------------------------------------------------
    def player_value(self, player_id: Optional[str] = None,
                     now: Optional[float] = None) -> Dict:
        now = time.time() if now is None else float(now)
        total_revenue = sum(t["amount"] for t in self._history)
        num_players = len(self._players)
        arpu = total_revenue / num_players if num_players else 0.0
        if player_id is None:
            return {
                "arpu": arpu,
                "total_revenue": total_revenue,
                "player_count": num_players,
            }
        analysis = self.behavior_analysis(player_id, now)
        player = self._players.get(player_id)
        first_seen = player["first_seen"] if player and player["first_seen"] is not None else now
        months_active = max((now - first_seen) / SECONDS_PER_DAY / 30.0, 1.0)
        monthly_spend = analysis["total_spent"] / months_active
        churn_risk = self.churn_prediction(player_id, now)["risk"]
        expected_months = self.config.ltv_horizon_months * (1.0 - churn_risk)
        ltv = analysis["total_spent"] + monthly_spend * expected_months
        return {
            "player_id": player_id,
            "ltv": ltv,
            "arpu": arpu,
            "total_spent": analysis["total_spent"],
            "monthly_spend": monthly_spend,
            "expected_remaining_months": expected_months,
            "churn_risk": churn_risk,
        }

    # ------------------------------------------------------------------
    # 流失预测：风险 + 预警
    # ------------------------------------------------------------------
    def churn_prediction(self, player_id: str, now: Optional[float] = None) -> Dict:
        now = time.time() if now is None else float(now)
        player = self._players.get(player_id)
        if not player:
            return {
                "player_id": player_id,
                "risk": 0.5,
                "will_churn": False,
                "warning": False,
                "warning_message": None,
                "factors": {"inactivity": 0.5, "login_decline": 0.5, "spend_decline": 0.5},
            }
        events = list(player["logins"]) + [t["timestamp"] for t in player["transactions"]]
        last_activity = max(events) if events else now
        days_inactive = max(0.0, (now - last_activity) / SECONDS_PER_DAY)
        inactivity = min(1.0, days_inactive / 30.0)
        recent_logins = sum(1 for ts in player["logins"] if now - ts <= 14 * SECONDS_PER_DAY)
        login_decline = 1.0 - min(1.0, recent_logins / 7.0)
        txs = player["transactions"]
        if txs:
            days_since_purchase = (now - max(t["timestamp"] for t in txs)) / SECONDS_PER_DAY
            spend_decline = min(1.0, days_since_purchase / 14.0)
        else:
            spend_decline = 0.5
        risk = round(min(1.0, max(0.0,
            0.5 * inactivity + 0.3 * login_decline + 0.2 * spend_decline)), 4)
        warning = risk >= self.config.churn_warning_threshold
        return {
            "player_id": player_id,
            "risk": risk,
            "will_churn": risk >= 0.5,
            "warning": warning,
            "warning_message": ("玩家流失风险过高，请及时干预" if warning else None),
            "factors": {
                "inactivity": inactivity,
                "login_decline": login_decline,
                "spend_decline": spend_decline,
            },
        }
