from app.tagging.search import extract_api_path_tokens


def test_extract_api_path_tokens_last_segment():
    toks = extract_api_path_tokens("quota/postiveAdjustApply 正向调额完整链路")
    assert "postiveAdjustApply" in toks
    assert "quota" in toks


def test_lexical_main_entry_disabled():
    """Product agent must not hard-map business nouns → Class#method."""
    from app.agent.explore_agent import _lexical_main_entry

    q = (
        "credit/creditQuery 授信查询接口做什么？\n"
        "（需求复述：合作方授信通知 creditNotification 同步额度）"
    )
    assert _lexical_main_entry(q) is None
