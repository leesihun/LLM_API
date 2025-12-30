"""
Nanocoder Configuration Generator
Automatically generates .nanocoder/config.json based on config.py settings
"""
import json
from pathlib import Path
import config


def generate_nanocoder_config():
    """
    Generate nanocoder config.json based on LLM backend settings

    Creates provider configuration for Ollama or llama.cpp based on config.LLM_BACKEND
    """
    config_dir = config.NANOCODER_CONFIG_DIR
    config_file = config_dir / "config.json"

    # Build provider configuration based on LLM backend
    providers = []

    if config.LLM_BACKEND in ["ollama", "auto"]:
        # Add Ollama provider
        ollama_base_url = config.OLLAMA_HOST.replace("/api", "").rstrip("/") + "/v1"

        providers.append({
            "name": "Ollama",
            "baseUrl": ollama_base_url,
            "models": [config.OLLAMA_MODEL]
        })

    if config.LLM_BACKEND in ["llamacpp", "auto"]:
        # Add llama.cpp provider
        providers.append({
            "name": "Llama.cpp",
            "baseUrl": config.LLAMACPP_HOST,
            "models": [config.LLAMACPP_MODEL]
        })

    # Build full config
    nanocoder_config = {
        "providers": providers,
        "defaultProvider": providers[0]["name"] if providers else "Ollama",
        "defaultModel": providers[0]["models"][0] if providers else config.OLLAMA_MODEL
    }

    # Write config file
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(nanocoder_config, f, indent=2)

    print(f"[NANOCODER CONFIG] Generated config at: {config_file}")
    print(f"[NANOCODER CONFIG] Provider: {nanocoder_config['defaultProvider']}")
    print(f"[NANOCODER CONFIG] Model: {nanocoder_config['defaultModel']}")

    return config_file


def ensure_nanocoder_config():
    """
    Ensure nanocoder config exists, create if missing

    Returns:
        Path to config file
    """
    config_file = config.NANOCODER_CONFIG_DIR / "config.json"

    if not config_file.exists():
        print(f"[NANOCODER CONFIG] Config file not found, generating...")
        return generate_nanocoder_config()

    print(f"[NANOCODER CONFIG] Using existing config: {config_file}")
    return config_file


if __name__ == "__main__":
    # Test config generation
    print("Generating nanocoder config...")
    config_path = generate_nanocoder_config()
    print(f"\nConfig generated at: {config_path}")

    # Print config content
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    print("\nConfig content:")
    print(json.dumps(config_data, indent=2))
