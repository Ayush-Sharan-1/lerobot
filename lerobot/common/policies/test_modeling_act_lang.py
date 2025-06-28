def smoke_test():
    """Minimal test to check basic functionality"""
    try:
        from lerobot.common.policies.act.modeling_act_lang import ACT, ACTPolicy
        from lerobot.common.policies.act.configuration_act import ACTConfig
        import torch
        
        config = ACTConfig()
        model = ACT(config)
        print("✅ Model creation successful")
        
        # Test task embedding creation
        embeddings = model.create_task_embeddings()
        print(f"✅ Task embeddings shape: {embeddings.shape}")
        
        # Test prepare_language with simple input
        batch = {"task": "place at grid spot 5", "observation.state": torch.randn(1, 14)}
        lang_embed = model.prepare_language(batch)
        print(f"✅ Language preparation successful: {lang_embed.shape}")
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    smoke_test()