"""
游戏内玩家经济行为分析与消费预测系统 - 验收测试
运行方式：python -m pytest tests/test_economic_behavior.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.economic_behavior import EconomicBehavior

NOW = 1_700_000_000.0
DAY = 86400.0
HOUR = 3600.0


def make_active_payer(system, player_id="payer", now=NOW):
    """近 30 天内规律消费、近期活跃的付费玩家。"""
    for i in range(6):
        ts = now - (25 - i * 5) * DAY
        system.record_login(player_id, timestamp=ts)
        system.record_transaction(player_id, 50.0 + i * 10.0,
                                  timestamp=ts, category="gems")
    system.record_login(player_id, timestamp=now - HOUR)
    return player_id


def make_inactive_player(system, player_id="churner", now=NOW):
    """40 天未活跃的流失边缘玩家。"""
    system.record_login(player_id, timestamp=now - 40 * DAY)
    system.record_transaction(player_id, 10.0, timestamp=now - 40 * DAY)
    return player_id


class TestEconomicBehavior:
    def setup_method(self):
        self.system = EconomicBehavior()

    # ---- 经济行为分析：消费模式 + 消费习惯 ----
    def test_case_01(self):
        pid = make_active_payer(self.system)
        result = self.system.behavior_analysis(pid, now=NOW)
        assert result["consumption_pattern"] in (
            "regular_spender", "occasional_spender", "high_roller")
        assert result["consumption_pattern"] == "regular_spender"

    def test_case_02(self):
        pid = make_active_payer(self.system)
        result = self.system.behavior_analysis(pid, now=NOW)
        habits = result["consumption_habits"]
        assert habits["favorite_category"] == "gems"
        assert habits["purchases_per_week"] > 0
        assert habits["spending_trend"] == "increasing"

    def test_case_03(self):
        pid = make_active_payer(self.system)
        result = self.system.behavior_analysis(pid, now=NOW)
        assert result["transaction_count"] == 6
        assert result["total_spent"] == pytest.approx(50+60+70+80+90+100)
        assert result["avg_amount"] == pytest.approx(75.0)

    # ---- 消费预测：会不会花钱 + 特征 + 模型 ----
    def test_case_04(self):
        pid = make_active_payer(self.system)
        result = self.system.consumption_prediction(pid, now=NOW)
        assert isinstance(result["will_spend"], bool)
        assert result["will_spend"] is True
        assert 0.0 <= result["probability"] <= 1.0

    def test_case_05(self):
        pid = make_active_payer(self.system)
        result = self.system.consumption_prediction(pid, now=NOW)
        features = result["features"]
        for key in ("transaction_count", "total_spent", "avg_amount",
                    "days_since_last_purchase", "purchase_frequency"):
            assert key in features
        assert features["transaction_count"] == 6
        assert features["total_spent"] > 0

    def test_case_06(self):
        pid = make_active_payer(self.system)
        samples, labels = [], []
        for i in range(20):
            samples.append({"transaction_count": 5, "total_spent": 200.0,
                            "avg_amount": 40.0, "days_since_last_purchase": 1.0,
                            "purchase_frequency": 3.0, "active_days": 20})
            labels.append(1)
            samples.append({"transaction_count": 0, "total_spent": 0.0,
                            "avg_amount": 0.0, "days_since_last_purchase": 300.0,
                            "purchase_frequency": 0.0, "active_days": 0})
            labels.append(0)
        info = self.system.train(samples, labels)
        assert info["trained"] is True
        result = self.system.consumption_prediction(pid, now=NOW)
        assert result["model"]["type"] == "logistic_regression"
        assert result["model"]["trained"] is True
        assert result["will_spend"] is True

    # ---- 付费意愿：评估 + 分层 ----
    def test_case_07(self):
        pid = make_active_payer(self.system)
        result = self.system.willingness_to_pay(pid, now=NOW)
        assert 0.0 <= result["wtp_score"] <= 100.0
        assert result["estimated_willingness_to_pay"] > 0
        assert set(result["evaluation"]) == {
            "spend_score", "frequency_score", "recency_score", "trend_score"}

    def test_case_08(self):
        whale = "whale"
        for i in range(10):
            ts = NOW - (20 - i * 2) * DAY
            self.system.record_transaction(whale, 300.0 + i * 50.0,
                                           timestamp=ts, category="gems")
        result = self.system.willingness_to_pay(whale, now=NOW)
        assert result["tier"] == "whale"
        free = self.system.willingness_to_pay("free_player", now=NOW)
        assert free["tier"] == "free"

    def test_case_09(self):
        heavy = make_active_payer(self.system, "heavy")
        self.system.record_transaction("light", 5.0, timestamp=NOW - 25 * DAY)
        heavy_wtp = self.system.willingness_to_pay(heavy, now=NOW)
        light_wtp = self.system.willingness_to_pay("light", now=NOW)
        assert heavy_wtp["wtp_score"] > light_wtp["wtp_score"]
        assert heavy_wtp["estimated_willingness_to_pay"] > \
            light_wtp["estimated_willingness_to_pay"]

    # ---- 玩家价值：LTV + ARPU ----
    def test_case_10(self):
        pid = make_active_payer(self.system)
        result = self.system.player_value(pid, now=NOW)
        assert result["ltv"] > 0
        assert result["ltv"] >= result["total_spent"]
        assert result["expected_remaining_months"] > 0

    def test_case_11(self):
        make_active_payer(self.system, "p1")
        make_active_payer(self.system, "p2")
        result = self.system.player_value(now=NOW)
        expected_arpu = (450.0 * 2) / 2
        assert result["arpu"] == pytest.approx(expected_arpu)
        assert result["player_count"] == 2
        per_player = self.system.player_value("p1", now=NOW)
        assert per_player["arpu"] == pytest.approx(expected_arpu)

    def test_case_12(self):
        payer = make_active_payer(self.system, "payer")
        self.system.record_login("nonpayer", timestamp=NOW - HOUR)
        payer_value = self.system.player_value(payer, now=NOW)
        nonpayer_value = self.system.player_value("nonpayer", now=NOW)
        assert payer_value["ltv"] > nonpayer_value["ltv"]

    # ---- 流失预测：风险 + 预警 ----
    def test_case_13(self):
        pid = make_inactive_player(self.system)
        result = self.system.churn_prediction(pid, now=NOW)
        assert 0.0 <= result["risk"] <= 1.0
        assert result["risk"] > 0.5
        assert result["will_churn"] is True
        assert set(result["factors"]) == {
            "inactivity", "login_decline", "spend_decline"}

    def test_case_14(self):
        pid = make_inactive_player(self.system)
        result = self.system.churn_prediction(pid, now=NOW)
        assert result["warning"] is True
        assert result["warning_message"]

    def test_case_15(self):
        pid = make_active_payer(self.system)
        result = self.system.churn_prediction(pid, now=NOW)
        assert result["risk"] < 0.5
        assert result["will_churn"] is False
        assert result["warning"] is False
        assert result["warning_message"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
