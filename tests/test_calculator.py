from app.tools.calculator import calculator

def run(expr: str) -> str:
    return calculator.invoke({"expression": expr})

def test_basic_arithmetic():
    assert "240" in run("(100 - 20) * 3")

def test_sqrt():
    assert "4" in run("sqrt(16)")

def test_pi_constant():
    assert "3.14" in run("pi")

def test_reject_import():
    result = run("__import__('os').system('ls')")
    assert "失败" in result or "未允许" in result

def test_reject_dunder():
    result = run("(1).__class__")
    assert "失败" in result or "未允许" in result

def test_reject_variable():
    result = run("x + 1")
    assert "失败" in result or "未允许" in result

def test_invalid_syntas():
    result = run("2 +* 3")
    assert "失败" in result