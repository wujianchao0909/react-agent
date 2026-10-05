from fastapi import Request

def get_agent(request: Request):
    """从 app.state 里拿到 lifespan 阶段构建好的 agent"""
    return request.app.state.agent