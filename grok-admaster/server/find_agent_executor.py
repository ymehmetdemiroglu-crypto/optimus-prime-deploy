
import os
import langchain

package_dir = os.path.dirname(langchain.__file__)
print(f"Searching in: {package_dir}")

for root, dirs, files in os.walk(package_dir):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "class AgentExecutor" in content:
                    print(f"Found AgentExecutor in: {filepath}")
