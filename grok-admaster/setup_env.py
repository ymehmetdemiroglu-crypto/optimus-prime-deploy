
import os
import sys

def main():
    print("---------------------------------------------------------")
    print("   Optimus Pryme - Environment Setup Wizard")
    print("---------------------------------------------------------")
    print("This script will help you configure the API keys for the system.")
    print("The keys will be saved to the .env file.")
    print("")

    # Load existing .env if present
    env_path = ".env"
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, val = line.strip().split("=", 1)
                    env_vars[key] = val

    # Helper to ask for input
    def ask_key(key_name, description, required=False):
        current = env_vars.get(key_name, "")
        if current:
            print(f"Current {key_name}: {current[:4]}...{current[-4:]} (configured)")
            prompt = f"Enter new {key_name} (or press Enter to keep current): "
        else:
            prompt = f"Enter {key_name} ({description}): "
        
        val = input(prompt).strip()
        if not val and current:
            return current
        return val

    # 1. OpenRouter (AI Models)
    print("\n[AI / LLM CONFIGURATION]")
    print("We recommend OpenRouter for pay-as-you-go access to GPT-4o, Claude 3.5, and DeepSeek.")
    openrouter_key = ask_key("OPENROUTER_API_KEY", "sk-or-v1-...", required=True)
    if openrouter_key:
        env_vars["OPENROUTER_API_KEY"] = openrouter_key

    # 2. DataForSEO (Market Intelligence)
    print("\n[MARKET INTELLIGENCE CONFIGURATION]")
    print("DataForSEO is recommended for pay-as-you-go Amazon product data and SEO stats.")
    dfs_login = ask_key("DATAFORSEO_LOGIN", "Your Login/Email")
    if dfs_login:
        env_vars["DATAFORSEO_LOGIN"] = dfs_login
    
    dfs_pass = ask_key("DATAFORSEO_PASSWORD", "Your API Password")
    if dfs_pass:
        env_vars["DATAFORSEO_PASSWORD"] = dfs_pass

    # SAVE
    print("\n---------------------------------------------------------")
    print("Saving configuration...")
    
    with open(env_path, "w") as f:
        for key, val in env_vars.items():
            f.write(f"{key}={val}\n")
    
    print(f"Configuration saved to {os.path.abspath(env_path)}")
    print("You can now restart the server.")

if __name__ == "__main__":
    main()
