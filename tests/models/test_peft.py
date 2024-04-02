from transformers import AutoModelForCausalLM
def test_peft():
    try:
        import peft
        from peft import get_peft_model
        
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=32,
            lora_dropout=0.05
        )
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
        
        lora_model = get_peft_model(model, lora_config)
        lora_model.print_trainable_parameters()

        print("Running PEFT is successful!")
    
    except:
        raise Exception("Sorry, peft is not installed")
