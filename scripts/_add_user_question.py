#!/usr/bin/env python
"""Add realistic `user_question`（用户原始表述）to every battery scenario."""
import json
import io

PATH = "data/log_diag_battery/scenarios.json"

USER_QUESTION = {
    "S01": "xxl-job 放款批次任务执行报错 handleCode=500，提示执行参数不能为空",
    "S02": "xxl-job 批量记账任务明明配了产品参数 productCodes，执行还是报「配置产品参数不能为空」",
    "S03": "xxl-job 批次任务传了空字符串参数 count/date 却显示成功，怀疑参数校验有问题",
    "S04": "MQ 消费还款状态消息报 JSON 解析失败，但消息却被直接签收不见了（疑似丢消息）",
    "S05": "MQ 收到还款明细为空被当成脏数据直接签收，疑似数据被静默丢弃",
    "S06": "MQ 消费线程里调 xxx-service 报域名解析失败 UnknownHostException，整个消费被影响",
    "S07": "SQL 报错 Unknown column 'activity_no'，线上表结构和代码对不上，放款活动业务挂了",
    "S08": "结清证明下载接口报「结清证明id不可为空」，参数校验不过",
    "S09": "风控调额接口空指针 NullPointerException，怀疑查不到客户信用记录",
    "S10": "利率调整校验报 NumberFormatException：For input string: \"null\"，怀疑参数缺失",
    "S11": "xxl-job 任务日志显示 JobThread toStop、容器销毁，是否异常",
    "S12": "MQ 消费获取 Redisson 锁失败后消息被 basicReject 重新投递，会不会丢消息或重复处理",
    "S13": "配置了放款接口不走风控，但实际却走了（新）风控引擎，为什么",
    "S14": "配置了调额走风控（isRisk），但日志显示「不过风控」直接落库，为什么",
    "S15": "配置了放款走新流程，但决断结果全 false 走了老流程，为什么",
}

with io.open(PATH, encoding="utf-8") as f:
    data = json.load(f)

for sc in data["scenarios"]:
    uq = USER_QUESTION.get(sc["id"], sc.get("user_question", ""))
    sc["user_question"] = uq

with io.open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"done: {len(data['scenarios'])} scenarios updated")