"""
Application controls go here
"""
from decimal import Decimal
from pgu import gui


# noinspection PyBroadException
def attr_changer(attr):
    ctx, solver_cls, prop, field, converter = attr

    solver = ctx.solver

    if solver:
        if not hasattr(solver, prop):
            raise ValueError('Solver %s has not attribute %s' % (solver, prop))

        try:
            value = converter(field.value)
            setattr(solver, prop, value)
        except Exception as e:
            print('Exception while setting prop %s to value %s. (%s)'
                  % (prop, field.value, e))


def create_attribute_field(ctx, solver, attribute):
    # create the control
    converter = attribute.converter
    name = attribute.name
    prop = attribute.prop
    default = attribute.default
    if converter == bool:
        field = gui.Switch(value=default)
    elif converter in (int, float, Decimal):
        field = gui.Input(value=default, size=10, align=1)
    else:
        field = gui.Input(value=default, size=10, align=-1)
    label = gui.Label(name)
    field.connect(gui.CHANGE,
                  attr_changer,
                  (ctx, solver, prop, field, converter))
    return label, field


def create_attribute_fields(ctx, solver):
    attributes = getattr(solver, 'ATTRIBUTES', [])
    for attribute in attributes:
        yield create_attribute_field(ctx, solver, attribute)


class Application(object):
    """
    Small wrapper around the pgu application
    """

    def __init__(self, ctx, screen_width, screen_height,
                 x, y, width, height):

        self._ctx = ctx
        self._solvers = ctx.solvers
        self._solver_label, self._solver_field = None, None
        self._solver_ui = None
        self._width = width
        self._height = height

        self._outer_container = gui.Container(width=screen_width,
                                              height=screen_height,
                                              align=-1, valign=-1)

        # outer container where stuff goes
        self._main_container = gui.Table(width=width, height=height,
                                         hpadding=5, vpadding=5,
                                         align=-1, valign=-1)

        self._outer_container.add(self._main_container, x, y)

        self._build_ui()

        application = gui.App()
        application.init(self._outer_container)

        self._application = application

    def _get_default_solver(self):
        solvers = list(sorted(self._solvers.values(),
                       key=lambda x: x.NAME))
        if solvers:
            return solvers[0]
        raise ValueError('No default solver found')

    def _build_solver_ui(self, solver):
        table = gui.Table(width=self._width, hpadding=5, vpadding=5,)

        # create the fields provided by the solver
        fields = list(create_attribute_fields(self._ctx, solver))

        for label, field in fields:
            table.tr()
            table.td(label, align=-1)
            table.td(field, align=-1)

        return table

    def _get_solver(self, name):
        solver = self._solvers.get(name)
        if not solver:
            raise ValueError('No solver of type %s' % name)
        return solver

    def _on_solver_change(self, field):
        value = field.value
        solver = self._get_solver(value)
        self._solver_container.widget = self._build_solver_ui(self._ctx, solver)

    def _create_solver_select(self):
        select = gui.Select()
        label = gui.Label('Choose Solver')
        for _, solver in self._solvers.items():
            select.add(solver.NAME, solver.TYPE)
        select.connect(gui.CHANGE, self._on_solver_change, select)
        return label, select

    def _build_buttons(self):
        self._start_button = gui.Button('Start')
        self._stop_button = gui.Button('Stop')
        self._reset_button = gui.Button('Reset')
        self._button_panel = gui.Document()
        self._button_panel.add(self._start_button)
        self._button_panel.add(self._stop_button)
        self._button_panel.add(self._reset_button)

    def _build_ui(self):
        self._solver_label, self._solver_field = self._create_solver_select()
        self._solver_ui = self._build_solver_ui(self._get_default_solver())
        self._solver_container = gui.ScrollArea(self._solver_ui)

        self._build_buttons()

        self._main_container.tr()
        self._main_container.td(self._solver_label, valign=-1, align=-1)
        self._main_container.td(self._solver_field, valign=-1, align=-1)

        self._main_container.tr()
        self._main_container.td(self._solver_container, colspan=2,
                                valign=-1, align=-1)

        self._main_container.tr()
        self._main_container.td(self._button_panel,
                                colspan=2)

    @property
    def app(self):
        return self._application


def create_application(ctx, screen_width, screen_height,
                       x, y, width, height):
    """
    Create a pgu applicaiton.
    """

    return Application(ctx, screen_width, screen_height,
                       x, y, width, height)


class ApplicationContext(object):
    def __init__(self, solvers, solver=None):
        self.solver = solver
        self.solvers = solvers