from guidance import capture, models, one_or_more, select, guidance


def test_capture():
    model = models.Mock()
    try:
        model += "This is" + capture(select(options=["bad", "quite bad"]), name="my_var")
        assert model["my_var"] in ["bad", "quite bad"]
    finally:
        model.close()


def test_capture_star():
    lm = models.Mock(b"<s>1234233234<s>")
    try:
        grammar = capture(one_or_more(select(["1", "2"])), name="test")
        lm2 = lm + grammar
        assert lm2["test"] == "12"
    finally:
        lm.close()

def test_capture_raw_function():
    lm = models.Mock(b"<s>1234233234<s>")
    try:
        lm += select(["1", "2"], name="state")

        @guidance
        def raw_fn(lm):
            if lm["state"] == "1":
                lm += select(["3", "4"], name="state_1")
            elif lm["state"] == "2":
                lm += select(["5", "6"], name="state_2")
            return lm
    
        lm_nocap = lm + "the beginning|" + raw_fn() + "|the end"
        lm_cap_arg = lm + "the beginning|" + capture("<cap>" + raw_fn() + "</cap>" , "cap_arg") + "|the end"
        lm_cap_kwarg = lm + "the beginning|" + capture("<cap>" + raw_fn() + "</cap>", name="cap_kwarg") + "|the end"
    
        # Bunch of random tests
        assert "state_1" in lm_nocap or "state_2" in lm_nocap
        assert "cap_arg" in lm_cap_arg
        assert "cap_kwarg" in lm_cap_kwarg
        assert lm_cap_arg["cap_arg"].startswith("<cap>")
        assert lm_cap_arg["cap_arg"].endswith("</cap>")
        assert lm_cap_kwarg["cap_kwarg"].startswith("<cap>")
        assert lm_cap_kwarg["cap_kwarg"].endswith("</cap>")
        assert len(lm_cap_arg["cap_arg"]) == len(lm_cap_kwarg["cap_kwarg"])

        assert str(lm_nocap).endswith("|the end")
        assert str(lm_cap_arg).endswith("|the end")
        assert str(lm_cap_kwarg).endswith("|the end")
    finally:
        lm.close()
