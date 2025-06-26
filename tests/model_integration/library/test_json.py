from guidance import models, json as gen_json


def test_json_usage_smoke(selected_model: models.Model):
    lm = selected_model

    lm += "My D&D character's stats: " + gen_json(
        schema={
            "type": "object",
            "properties": {
                "strength": {"type": "integer", "minimum": 0.0, "maximum": 20},
                "agility": {"type": "integer", "minimum": 0.0, "maximum": 20},
                "intelligence": {"type": "integer", "minimum": 0.0, "maximum": 20},
                "endurance": {"type": "integer", "minimum": 0.0, "maximum": 20},
                "charisma": {"type": "integer", "minimum": 0.0, "maximum": 20},
                "luck": {"type": "integer", "minimum": 0.0, "maximum": 20},
                "wisdom": {"type": "integer", "minimum": 0.0, "maximum": 20},
            },
            "required": ["strength", "agility", "intelligence", "endurance", "charisma", "luck", "wisdom"],
            "additionalProperties": False,
        }
    )

    usage = lm._get_usage()

    # What follows are rough estimates of the token usage based on the schema.
    # Future devs: these might be blatantly wrong, so please adjust them if needed.

    n_props = 7  # strength, agility, intelligence, endurance, charisma, luck, wisdom

    prompt_lb = 4
    prompt_ub = 15

    ff_lb = 1 * n_props  # 1 per pr
    ff_ub = 10 * n_props + 4  # 10 tokens per property + 4 for the boundaries

    gen_lb = 1 * n_props  # 1 token per property
    gen_ub = 3 * n_props  # 3 tokens per property

    assert prompt_lb <= usage.input_tokens - usage.output_tokens - usage.cached_input_tokens <= prompt_ub
    assert ff_lb <= usage.ff_tokens <= ff_ub
    assert gen_lb <= usage.output_tokens - usage.ff_tokens <= gen_ub
    assert (ff_lb / (ff_lb + gen_ub)) <= usage.token_savings <= (ff_ub / (ff_ub + gen_lb))
