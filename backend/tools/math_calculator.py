"""
Math and Calculator Tool
Performs precise mathematical calculations using SymPy
"""

import logging
from typing import Any, Dict
import re


logger = logging.getLogger(__name__)


class MathCalculator:
    """
    Advanced mathematics and calculator tool

    Features:
    - Basic arithmetic
    - Algebraic equations
    - Calculus (derivatives, integrals)
    - Unit conversions
    - Symbolic mathematics
    """

    def __init__(self):
        try:
            import sympy
            self.sympy = sympy
            self.sympy_available = True
        except ImportError:
            logger.warning("SymPy not installed. Install with: pip install sympy")
            self.sympy_available = False

    async def calculate(self, expression: str, return_latex: bool = False) -> str | Dict[str, str]:
        """
        Evaluate a mathematical expression

        Args:
            expression: Math expression to evaluate
            return_latex: If True, return dict with both plain text and LaTeX

        Returns:
            Result as string, or dict with 'result' and 'latex' keys
        """
        logger.info(f"[Math Calculator] Calculating: {expression}")

        if not self.sympy_available:
            # Fall back to basic eval (limited and less safe)
            result = await self._basic_calculate(expression)
            return {"result": result, "latex": None} if return_latex else result

        try:
            # Clean the expression
            expression = expression.strip()

            # Handle special commands
            if "solve" in expression.lower():
                return await self._solve_equation(expression, return_latex)
            elif "derivative" in expression.lower() or "diff" in expression.lower():
                return await self._calculate_derivative(expression, return_latex)
            elif "integral" in expression.lower() or "integrate" in expression.lower():
                return await self._calculate_integral(expression, return_latex)
            elif "factor" in expression.lower():
                return await self._factor_expression(expression, return_latex)
            elif "expand" in expression.lower():
                return await self._expand_expression(expression, return_latex)
            elif "simplify" in expression.lower():
                return await self._simplify_expression(expression, return_latex)
            else:
                # Standard calculation
                return await self._evaluate_expression(expression, return_latex)

        except Exception as e:
            logger.error(f"[Math Calculator] Error: {e}")
            error_msg = f"Error calculating: {str(e)}"
            return {"result": error_msg, "latex": None} if return_latex else error_msg

    async def _evaluate_expression(self, expression: str, return_latex: bool = False) -> str | Dict[str, str]:
        """Evaluate a mathematical expression"""
        try:
            # Parse and evaluate
            result = self.sympy.sympify(expression)
            evaluated = self.sympy.N(result, 10)  # 10 decimal places

            # Format result
            if evaluated.is_integer:
                plain_text = f"Result: {int(evaluated)}"
            else:
                plain_text = f"Result: {float(evaluated):.10g}"

            if return_latex:
                latex_expr = self.sympy.latex(result)
                latex_result = self.sympy.latex(evaluated)
                latex_str = f"{latex_expr} = {latex_result}"
                return {"result": plain_text, "latex": latex_str}
            else:
                return plain_text

        except Exception as e:
            error_msg = f"Error evaluating expression: {str(e)}"
            return {"result": error_msg, "latex": None} if return_latex else error_msg

    async def _solve_equation(self, expression: str, return_latex: bool = False) -> str | Dict[str, str]:
        """Solve an equation"""
        try:
            # Extract equation (look for = sign)
            if "=" in expression:
                # Split by = and create equation
                parts = expression.split("=")
                if len(parts) == 2:
                    lhs = self.sympy.sympify(parts[0].strip())
                    rhs = self.sympy.sympify(parts[1].strip())
                    equation = self.sympy.Eq(lhs, rhs)

                    # Solve
                    solutions = self.sympy.solve(equation)

                    if solutions:
                        if isinstance(solutions, list):
                            solutions_str = ", ".join(str(sol) for sol in solutions)
                            plain_text = f"Solutions: {solutions_str}"

                            if return_latex:
                                latex_eq = self.sympy.latex(equation)
                                latex_sols = ", ".join(self.sympy.latex(sol) for sol in solutions)
                                latex_str = f"{latex_eq} \\quad \\Rightarrow \\quad x = {latex_sols}"
                                return {"result": plain_text, "latex": latex_str}
                            return plain_text
                        else:
                            plain_text = f"Solution: {solutions}"
                            if return_latex:
                                latex_eq = self.sympy.latex(equation)
                                latex_sol = self.sympy.latex(solutions)
                                latex_str = f"{latex_eq} \\quad \\Rightarrow \\quad {latex_sol}"
                                return {"result": plain_text, "latex": latex_str}
                            return plain_text
                    else:
                        msg = "No solutions found"
                        return {"result": msg, "latex": None} if return_latex else msg

            msg = "Please provide an equation with '=' sign"
            return {"result": msg, "latex": None} if return_latex else msg

        except Exception as e:
            error_msg = f"Error solving equation: {str(e)}"
            return {"result": error_msg, "latex": None} if return_latex else error_msg

    async def _calculate_derivative(self, expression: str, return_latex: bool = False) -> str | Dict[str, str]:
        """Calculate derivative"""
        try:
            # Extract function and variable
            # Pattern: "derivative of x^2 with respect to x"
            match = re.search(r'derivative of (.+?) with respect to (\w+)', expression, re.IGNORECASE)
            if match:
                func_str = match.group(1).strip()
                var_str = match.group(2).strip()

                func = self.sympy.sympify(func_str)
                var = self.sympy.Symbol(var_str)

                derivative = self.sympy.diff(func, var)

                plain_text = f"Derivative: {derivative}"

                if return_latex:
                    latex_func = self.sympy.latex(func)
                    latex_var = self.sympy.latex(var)
                    latex_deriv = self.sympy.latex(derivative)
                    latex_str = f"\\frac{{d}}{{d{latex_var}}} \\left({latex_func}\\right) = {latex_deriv}"
                    return {"result": plain_text, "latex": latex_str}
                return plain_text

            msg = "Please specify: 'derivative of [function] with respect to [variable]'"
            return {"result": msg, "latex": None} if return_latex else msg

        except Exception as e:
            error_msg = f"Error calculating derivative: {str(e)}"
            return {"result": error_msg, "latex": None} if return_latex else error_msg

    async def _calculate_integral(self, expression: str, return_latex: bool = False) -> str | Dict[str, str]:
        """Calculate integral"""
        try:
            # Extract function and variable
            match = re.search(r'integral of (.+?) with respect to (\w+)', expression, re.IGNORECASE)
            if match:
                func_str = match.group(1).strip()
                var_str = match.group(2).strip()

                func = self.sympy.sympify(func_str)
                var = self.sympy.Symbol(var_str)

                integral = self.sympy.integrate(func, var)

                plain_text = f"Integral: {integral} + C"

                if return_latex:
                    latex_func = self.sympy.latex(func)
                    latex_var = self.sympy.latex(var)
                    latex_integral = self.sympy.latex(integral)
                    latex_str = f"\\int {latex_func} \\, d{latex_var} = {latex_integral} + C"
                    return {"result": plain_text, "latex": latex_str}
                return plain_text

            msg = "Please specify: 'integral of [function] with respect to [variable]'"
            return {"result": msg, "latex": None} if return_latex else msg

        except Exception as e:
            error_msg = f"Error calculating integral: {str(e)}"
            return {"result": error_msg, "latex": None} if return_latex else error_msg

    async def _factor_expression(self, expression: str, return_latex: bool = False) -> str | Dict[str, str]:
        """Factor an expression"""
        try:
            # Extract expression after "factor"
            match = re.search(r'factor\s+(.+)', expression, re.IGNORECASE)
            if match:
                expr_str = match.group(1).strip()
                expr = self.sympy.sympify(expr_str)
                factored = self.sympy.factor(expr)

                plain_text = f"Factored form: {factored}"

                if return_latex:
                    latex_expr = self.sympy.latex(expr)
                    latex_factored = self.sympy.latex(factored)
                    latex_str = f"{latex_expr} = {latex_factored}"
                    return {"result": plain_text, "latex": latex_str}
                return plain_text

            msg = "Please specify: 'factor [expression]'"
            return {"result": msg, "latex": None} if return_latex else msg

        except Exception as e:
            error_msg = f"Error factoring: {str(e)}"
            return {"result": error_msg, "latex": None} if return_latex else error_msg

    async def _expand_expression(self, expression: str, return_latex: bool = False) -> str | Dict[str, str]:
        """Expand an expression"""
        try:
            match = re.search(r'expand\s+(.+)', expression, re.IGNORECASE)
            if match:
                expr_str = match.group(1).strip()
                expr = self.sympy.sympify(expr_str)
                expanded = self.sympy.expand(expr)

                plain_text = f"Expanded form: {expanded}"

                if return_latex:
                    latex_expr = self.sympy.latex(expr)
                    latex_expanded = self.sympy.latex(expanded)
                    latex_str = f"{latex_expr} = {latex_expanded}"
                    return {"result": plain_text, "latex": latex_str}
                return plain_text

            msg = "Please specify: 'expand [expression]'"
            return {"result": msg, "latex": None} if return_latex else msg

        except Exception as e:
            error_msg = f"Error expanding: {str(e)}"
            return {"result": error_msg, "latex": None} if return_latex else error_msg

    async def _simplify_expression(self, expression: str, return_latex: bool = False) -> str | Dict[str, str]:
        """Simplify an expression"""
        try:
            match = re.search(r'simplify\s+(.+)', expression, re.IGNORECASE)
            if match:
                expr_str = match.group(1).strip()
                expr = self.sympy.sympify(expr_str)
                simplified = self.sympy.simplify(expr)

                plain_text = f"Simplified form: {simplified}"

                if return_latex:
                    latex_expr = self.sympy.latex(expr)
                    latex_simplified = self.sympy.latex(simplified)
                    latex_str = f"{latex_expr} = {latex_simplified}"
                    return {"result": plain_text, "latex": latex_str}
                return plain_text

            msg = "Please specify: 'simplify [expression]'"
            return {"result": msg, "latex": None} if return_latex else msg

        except Exception as e:
            error_msg = f"Error simplifying: {str(e)}"
            return {"result": error_msg, "latex": None} if return_latex else error_msg

    async def _basic_calculate(self, expression: str) -> str:
        """
        Basic calculator fallback when SymPy is not available
        """
        try:
            # Very limited and UNSAFE - only for basic arithmetic
            # Should install SymPy for production use
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error (SymPy not installed): {str(e)}\nInstall SymPy for advanced math: pip install sympy"


# Global instance
math_calculator = MathCalculator()
