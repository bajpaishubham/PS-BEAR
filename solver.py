from __future__ import annotations

import re

import sympy
import sympy.physics.units as u
from sympy.physics.units import convert_to


class FormulaParseError(ValueError):
    pass


# Names allowed when parsing the sympy_formula strings. Cross is registered as an
# undefined function so vector cross products parse without evaluating.
_NAMESPACE = {
    "Eq": sympy.Eq,
    "Derivative": sympy.Derivative,
    "Integral": sympy.Integral,
    "Abs": sympy.Abs,
    "Cross": sympy.Function("Cross"),
    "Dot": sympy.Function("Dot"),
    "sqrt": sympy.sqrt,
    "exp": sympy.exp,
    "log": sympy.log,
    "ln": sympy.log,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "pi": sympy.pi,
}

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# Unit catalog backed by sympy.physics.units, covering the physical quantities that
# appear across the formula table (length, time, mass, force, energy, power, pressure,
# charge, current, voltage, resistance, capacitance, inductance, magnetic field/flux,
# frequency, amount, luminous, plus the derived velocity/acceleration/density/field/...).
# Built defensively: any unit, prefix, or base missing in the installed sympy is skipped.
def _build_unit_catalog() -> dict[str, sympy.Expr]:
    catalog: dict[str, sympy.Expr] = {"dimensionless": sympy.Integer(1)}

    def base(name: str):
        return getattr(u, name, None)

    # Named SI / common units that exist as attributes of sympy.physics.units.
    simple = [
        # length
        "meter", "decimeter", "centimeter", "millimeter", "micrometer", "nanometer",
        "picometer", "kilometer", "angstrom", "mile", "foot", "inch", "yard",
        "astronomical_unit", "nautical_mile",
        # time
        "second", "millisecond", "microsecond", "nanosecond", "picosecond",
        "minute", "hour", "day", "year",
        # mass / amount
        "kilogram", "gram", "milligram", "microgram", "metric_ton",
        "atomic_mass_unit", "pound", "mole",
        # thermodynamic / luminous
        "kelvin", "candela", "lux",
        # mechanics: force, energy, power, pressure
        "newton", "joule", "electronvolt", "watt", "pascal", "bar", "atmosphere",
        "mmHg", "torr", "psi",
        # electromagnetism
        "ampere", "coulomb", "volt", "ohm", "farad", "henry", "siemens",
        "tesla", "weber", "hertz",
        # geometry / angle / volume
        "radian", "degree", "steradian", "liter", "milliliter", "deciliter",
    ]
    for name in simple:
        quantity = base(name)
        if quantity is not None:
            catalog[name] = quantity

    # Prefixed units built as (prefix * base) — handled correctly by convert_to.
    prefixed = [
        ("milliampere", "milli", "ampere"), ("microampere", "micro", "ampere"),
        ("millicoulomb", "milli", "coulomb"), ("microcoulomb", "micro", "coulomb"),
        ("nanocoulomb", "nano", "coulomb"),
        ("millivolt", "milli", "volt"), ("kilovolt", "kilo", "volt"), ("microvolt", "micro", "volt"),
        ("kiloohm", "kilo", "ohm"), ("megaohm", "mega", "ohm"),
        ("microfarad", "micro", "farad"), ("nanofarad", "nano", "farad"), ("picofarad", "pico", "farad"),
        ("millihenry", "milli", "henry"), ("microhenry", "micro", "henry"),
        ("millitesla", "milli", "tesla"), ("microtesla", "micro", "tesla"),
        ("millinewton", "milli", "newton"), ("kilonewton", "kilo", "newton"),
        ("millijoule", "milli", "joule"), ("kilojoule", "kilo", "joule"), ("megajoule", "mega", "joule"),
        ("kiloelectronvolt", "kilo", "electronvolt"), ("megaelectronvolt", "mega", "electronvolt"),
        ("milliwatt", "milli", "watt"), ("kilowatt", "kilo", "watt"), ("megawatt", "mega", "watt"),
        ("kilopascal", "kilo", "pascal"), ("megapascal", "mega", "pascal"),
        ("kilohertz", "kilo", "hertz"), ("megahertz", "mega", "hertz"), ("gigahertz", "giga", "hertz"),
        ("millimole", "milli", "mole"),
    ]
    for key, prefix_name, base_name in prefixed:
        prefix, quantity = base(prefix_name), base(base_name)
        if prefix is not None and quantity is not None:
            catalog[key] = prefix * quantity

    # Common units that have no named attribute, defined from a base unit.
    if base("newton") is not None:
        catalog["dyne"] = sympy.Rational(1, 10**5) * u.newton
    if base("joule") is not None:
        catalog["calorie"] = sympy.Float(4.184) * u.joule
        catalog["erg"] = sympy.Rational(1, 10**7) * u.joule
        catalog["kilowatt_hour"] = sympy.Integer(3_600_000) * u.joule
    if base("watt") is not None:
        catalog["horsepower"] = sympy.Float(745.6998715822702) * u.watt
    if base("tesla") is not None:
        catalog["gauss"] = sympy.Rational(1, 10**4) * u.tesla
    if base("candela") is not None and base("steradian") is not None:
        catalog["lumen"] = u.candela * u.steradian

    # Derived / compound units expressed from base units.
    meter, second = base("meter"), base("second")
    if meter is not None and second is not None:
        catalog["meter/second"] = meter / second
        catalog["meter/second**2"] = meter / second**2
    if base("kilometer") is not None and base("hour") is not None:
        catalog["kilometer/hour"] = u.kilometer / u.hour
    if base("centimeter") is not None and second is not None:
        catalog["centimeter/second"] = u.centimeter / second
    if base("radian") is not None and second is not None:
        catalog["radian/second"] = u.radian / second
        catalog["radian/second**2"] = u.radian / second**2
    if meter is not None:
        catalog["meter**2"] = meter**2
        catalog["meter**3"] = meter**3
    if base("centimeter") is not None:
        catalog["centimeter**2"] = u.centimeter**2
        catalog["centimeter**3"] = u.centimeter**3
    if base("kilogram") is not None and meter is not None:
        catalog["kilogram/meter**3"] = u.kilogram / meter**3
        if second is not None:
            catalog["kilogram*meter/second"] = u.kilogram * meter / second
    if base("gram") is not None and base("centimeter") is not None:
        catalog["gram/centimeter**3"] = u.gram / u.centimeter**3
    if base("volt") is not None and meter is not None:
        catalog["volt/meter"] = u.volt / meter
    if base("newton") is not None and base("coulomb") is not None:
        catalog["newton/coulomb"] = u.newton / u.coulomb
    if base("watt") is not None and meter is not None:
        catalog["watt/meter**2"] = u.watt / meter**2
    if base("pascal") is not None and second is not None:
        catalog["pascal*second"] = u.pascal * second

    return catalog


UNIT_CATALOG = _build_unit_catalog()


def unit_names() -> list[str]:
    return list(UNIT_CATALOG.keys())


def parse_sympy(formula_str: str):
    formula_str = (formula_str or "").strip()
    if not formula_str:
        raise FormulaParseError("Empty symbolic formula.")
    # Map every identifier to a Symbol except the known callables/constants. This
    # prevents names such as `beta` or `gamma` from resolving to SymPy special
    # functions in the default namespace.
    locals_dict = dict(_NAMESPACE)
    for name in set(_IDENTIFIER_RE.findall(formula_str)):
        if name not in locals_dict:
            locals_dict[name] = sympy.Symbol(name)
    try:
        return sympy.sympify(formula_str, locals=locals_dict)
    except Exception as exc:  # noqa: BLE001 - surface any parser failure uniformly
        raise FormulaParseError(f"Could not parse the symbolic formula: {exc}") from exc


def symbols_in(expr) -> list[sympy.Symbol]:
    return sorted(expr.free_symbols, key=lambda symbol: symbol.name)


def is_algebraic(expr) -> bool:
    """True when the expression has no calculus/vector operators, so it can be
    evaluated numerically by substitution."""
    if expr.atoms(sympy.Derivative, sympy.Integral):
        return False
    if expr.atoms(sympy.core.function.AppliedUndef):  # e.g. Cross(...), Dot(...)
        return False
    return True


def solve_analytical(equation, target: sympy.Symbol) -> list:
    """Solve an Eq (or expression assumed equal to zero) for the target symbol."""
    if isinstance(equation, sympy.Equality):
        relation = equation
    else:
        relation = sympy.Eq(equation, 0)
    try:
        solutions = sympy.solve(relation, target, dict=False)
    except Exception as exc:  # noqa: BLE001
        raise FormulaParseError(f"Could not solve for {target}: {exc}") from exc
    if not isinstance(solutions, list):
        solutions = [solutions]
    return solutions


def to_latex(expr) -> str:
    return sympy.latex(expr)


def solve_numerical(
    solution_expr,
    value_unit_map: dict[sympy.Symbol, tuple[float, str]],
    output_unit_name: str,
) -> tuple[float, str]:
    """Substitute (value, unit) pairs into a solution expression and convert the
    result into the requested output unit.

    value_unit_map maps each input symbol to (numeric_value, unit_name).
    Returns (magnitude_in_output_unit, pretty_result_string).
    """
    substitutions = {}
    for symbol, (value, unit_name) in value_unit_map.items():
        unit_quantity = UNIT_CATALOG.get(unit_name, sympy.Integer(1))
        substitutions[symbol] = sympy.Float(value) * unit_quantity

    result = solution_expr.subs(substitutions)
    output_unit = UNIT_CATALOG.get(output_unit_name, sympy.Integer(1))

    converted = convert_to(result, output_unit)
    try:
        magnitude = float(sympy.N(converted / output_unit))
    except (TypeError, ValueError) as exc:
        raise FormulaParseError(
            f"Result units are not compatible with '{output_unit_name}'. "
            f"Computed: {sympy.N(result)}"
        ) from exc

    label = output_unit_name if output_unit_name != "dimensionless" else ""
    pretty = f"{magnitude:.6g} {label}".strip()
    return magnitude, pretty
