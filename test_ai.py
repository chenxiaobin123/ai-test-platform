from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载根目录下的.env配置文件
load_dotenv()

# 打印配置信息（方便调试）
print("正在加载配置...")
print(f"API_KEY: {os.getenv('OPENAI_API_KEY')[:10]}...")
print(f"BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
print(f"MODEL: {os.getenv('AI_MODEL')}")
print("-" * 50)

try:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    response = client.chat.completions.create(
        model=os.getenv("AI_MODEL"),
        messages=[{"role": "user", "content": "你好，帮我生成一个简单的百度搜索测试用例"}],
        max_tokens=200,
        temperature=0
    )

    print("✅ 火山引擎API调用成功！")
    print("=" * 50)
    print("模型回复：")
    print(response.choices[0].message.content)
    print("=" * 50)
    print("\n🎉 恭喜！AI配置完全正常，可以重启平台使用AI功能了！")

except Exception as e:
    print("❌ API调用失败！")
    print(f"错误信息：{str(e)}")
    print("\n请检查：")
    print("1. .env文件中的OPENAI_API_KEY是否正确")
    print("2. AI_MODEL是否填的是ep-开头的Endpoint ID")
    print("3. 网络是否能正常访问ark.cn-beijing.volces.com")