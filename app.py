

import ast
import math
import operator

from flask import Flask, jsonify, render_template, request  # type: ignore[import-not-found]

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Safe expression evaluator (no eval() - parses the expression with AST)
# ---------------------------------------------------------------------------

BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

MAX_EXPONENT = 1000
MAX_FACTORIAL = 170
MAX_EXPR_LEN = 300


class CalcError(Exception):
    """Raised for any user-facing calculation error."""


def build_functions(angle_mode: str) -> dict:
    """Return allowed functions. Trig functions respect DEG / RAD mode."""
    deg = angle_mode == "deg"

    def to_rad(x):
        return math.radians(x) if deg else x

    def from_rad(x):
        return math.degrees(x) if deg else x

    def factorial(x):
        if x < 0 or x != int(x):
            raise CalcError("Factorial needs a non-negative whole number")
        if x > MAX_FACTORIAL:
            raise CalcError(f"Factorial limit is {MAX_FACTORIAL}")
        return math.factorial(int(x))

    def sqrt(x):
        if x < 0:
            raise CalcError("Square root of a negative number")
        return math.sqrt(x)

    def log(x):
        if x <= 0:
            raise CalcError("Log needs a positive number")
        return math.log10(x)

    def ln(x):
        if x <= 0:
            raise CalcError("ln needs a positive number")
        return math.log(x)

    def tan(x):
        r = to_rad(x)
        if abs(math.cos(r)) < 1e-12:
            raise CalcError("tan is undefined here")
        return math.tan(r)

    def asin(x):
        if not -1 <= x <= 1:
            raise CalcError("asin needs a value between -1 and 1")
        return from_rad(math.asin(x))

    def acos(x):
        if not -1 <= x <= 1:
            raise CalcError("acos needs a value between -1 and 1")
        return from_rad(math.acos(x))

    return {
        "sin": lambda x: math.sin(to_rad(x)),
        "cos": lambda x: math.cos(to_rad(x)),
        "tan": tan,
        "asin": asin,
        "acos": acos,
        "atan": lambda x: from_rad(math.atan(x)),
        "sqrt": sqrt,
        "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
        "log": log,
        "ln": ln,
        "exp": math.exp,
        "abs": abs,
        "fact": factorial,
        "round": round,
    }


def evaluate(node, functions):
    """Recursively evaluate an AST node, allowing only safe operations."""
    if isinstance(node, ast.Expression):
        return evaluate(node.body, functions)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalcError("Only numbers are allowed")

    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise CalcError(f"Unknown name: {node.id}")

    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        left = evaluate(node.left, functions)
        right = evaluate(node.right, functions)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise CalcError("Cannot divide by zero")
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalcError("Exponent too large")
        return BIN_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](evaluate(node.operand, functions))

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in functions:
            raise CalcError(f"Unknown function: {name}")
        if node.keywords:
            raise CalcError("Invalid function call")
        args = [evaluate(a, functions) for a in node.args]
        return functions[name](*args)

    raise CalcError("Invalid expression")


def normalize(expr: str) -> str:
    """Convert display symbols into Python syntax."""
    replacements = {
        "×": "*",
        "÷": "/",
        "−": "-",
        "^": "**",
        "π": "pi",
        "√": "sqrt",
    }
    for old, new in replacements.items():
        expr = expr.replace(old, new)
    # percentage: 50% -> (50/100)
    expr = expr.replace("%", "/100")
    return expr


def format_result(value) -> str:
    """Make results look clean: no float noise, sensible precision."""
    if isinstance(value, complex):
        raise CalcError("Result is not a real number")
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            raise CalcError("Result is undefined")
        value = round(value, 10)
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        if abs(value) >= 1e15 or (0 < abs(value) < 1e-9):
            return f"{value:.6e}"
        return f"{value:.10g}"
    if isinstance(value, int) and abs(value) >= 1e15:
        return f"{value:.6e}"
    return str(value)


def calculate(expression: str, angle_mode: str = "deg") -> str:
    expression = (expression or "").strip()
    if not expression:
        raise CalcError("Empty expression")
    if len(expression) > MAX_EXPR_LEN:
        raise CalcError("Expression too long")

    expr = normalize(expression)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise CalcError("Syntax error")

    try:
        result = evaluate(tree, build_functions(angle_mode))
    except CalcError:
        raise
    except OverflowError:
        raise CalcError("Number too large")
    except (TypeError, ValueError):
        raise CalcError("Math error")

    return format_result(result)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/calculate", methods=["POST"])
def api_calculate():
    data = request.get_json(silent=True) or {}
    expression = data.get("expression", "")
    angle_mode = "rad" if data.get("angle_mode") == "rad" else "deg"
    try:
        result = calculate(expression, angle_mode)
        return jsonify({"ok": True, "result": result})
    except CalcError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True)