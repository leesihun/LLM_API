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

    async def calculate(self, expression: str) -> str:
        """
        Evaluate a mathematical expression

        Args:
            expression: Math expression to evaluate

        Returns:
            Result as string
        """
        logger.info(f"[Math Calculator] Calculating: {expression}")

        if not self.sympy_available:
            # Fall back to basic eval (limited and less safe)
            return await self._basic_calculate(expression)

        try:
            # Clean the expression
            expression = expression.strip()

            # Handle special commands
            if "solve" in expression.lower():
                return await self._solve_equation(expression)
            elif "derivative" in expression.lower() or "diff" in expression.lower():
                return await self._calculate_derivative(expression)
            elif "integral" in expression.lower() or "integrate" in expression.lower():
                return await self._calculate_integral(expression)
            elif "factor" in expression.lower():
                return await self._factor_expression(expression)
            elif "expand" in expression.lower():
                return await self._expand_expression(expression)
            elif "simplify" in expression.lower():
                return await self._simplify_expression(expression)
            else:
                # Standard calculation
                return await self._evaluate_expression(expression)

        except Exception as e:
            logger.error(f"[Math Calculator] Error: {e}")
            return f"Error calculating: {str(e)}"

    async def _evaluate_expression(self, expression: str) -> str:
        """Evaluate a mathematical expression"""
        try:
            # Parse and evaluate
            result = self.sympy.sympify(expression)
            evaluated = self.sympy.N(result, 10)  # 10 decimal places

            # Format result
            if evaluated.is_integer:
                return f"Result: {int(evaluated)}"
            else:
                return f"Result: {float(evaluated):.10g}"

        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    async def _solve_equation(self, expression: str) -> str:
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
                            return f"Solutions: {solutions_str}"
                        else:
                            return f"Solution: {solutions}"
                    else:
                        return "No solutions found"

            return "Please provide an equation with '=' sign"

        except Exception as e:
            return f"Error solving equation: {str(e)}"

    async def _calculate_derivative(self, expression: str) -> str:
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

                return f"Derivative: {derivative}"

            return "Please specify: 'derivative of [function] with respect to [variable]'"

        except Exception as e:
            return f"Error calculating derivative: {str(e)}"

    async def _calculate_integral(self, expression: str) -> str:
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

                return f"Integral: {integral} + C"

            return "Please specify: 'integral of [function] with respect to [variable]'"

        except Exception as e:
            return f"Error calculating integral: {str(e)}"

    async def _factor_expression(self, expression: str) -> str:
        """Factor an expression"""
        try:
            # Extract expression after "factor"
            match = re.search(r'factor\s+(.+)', expression, re.IGNORECASE)
            if match:
                expr_str = match.group(1).strip()
                expr = self.sympy.sympify(expr_str)
                factored = self.sympy.factor(expr)

                return f"Factored form: {factored}"

            return "Please specify: 'factor [expression]'"

        except Exception as e:
            return f"Error factoring: {str(e)}"

    async def _expand_expression(self, expression: str) -> str:
        """Expand an expression"""
        try:
            match = re.search(r'expand\s+(.+)', expression, re.IGNORECASE)
            if match:
                expr_str = match.group(1).strip()
                expr = self.sympy.sympify(expr_str)
                expanded = self.sympy.expand(expr)

                return f"Expanded form: {expanded}"

            return "Please specify: 'expand [expression]'"

        except Exception as e:
            return f"Error expanding: {str(e)}"

    async def _simplify_expression(self, expression: str) -> str:
        """Simplify an expression"""
        try:
            match = re.search(r'simplify\s+(.+)', expression, re.IGNORECASE)
            if match:
                expr_str = match.group(1).strip()
                expr = self.sympy.sympify(expr_str)
                simplified = self.sympy.simplify(expr)

                return f"Simplified form: {simplified}"

            return "Please specify: 'simplify [expression]'"

        except Exception as e:
            return f"Error simplifying: {str(e)}"

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
