#!/usr/bin/env python

import numpy as np
import numpy.random as npr
import numba as nb

import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim

from os import path
import neuro
from neuro import (
    a_m, b_m,
    a_n, b_n,
    a_h, b_h
)

from typing import Callable, Optional, Any
import abc


# ------------------------------------------------

PI = np.pi
TOLERANCE = 5e-4

# ------------------------------------------------


class NeuronFigure(abc.ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def plot(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def figure(self):
        raise NotImplementedError

    def set_defaults(self):
        self.titles  = None
        self.xlimits = None
        self.ylimits = None
        self.zlimits = None

    def set_titles(self, *args):
        assert(len(args) == len(self.axes)) #type: ignore
        self.titles = args

    def set_title(self, *args):
        self.set_titles(*args)

    def set_xlimits(self, *args):
        match args:
            case [float() | int(), float() | int()] if len(self.axes) == 1: #type: ignore
                self.xlimits = [args]
            case [tuple(), *_] if len(self.axes) == len(args): #type: ignore
                self.xlimits = args
            case _:
                raise ValueError(f'Invalid xlimits: {args}')

    def set_xlimit(self, *args):
        self.set_xlimits(*args)

    def set_ylimits(self, *args):
        match args:
            case [float() | int(), float() | int()] if len(self.axes) == 1: #type: ignore
                self.ylimits = [args]
            case [tuple(), *_] if len(self.axes) == len(args): #type: ignore
                self.ylimits = args
            case _:
                raise ValueError(f'Invalid ylimits: {args}')

    def set_ylimit(self, *args):
        self.set_ylimits(*args)

    def set_zlimits(self, *args):
        match args:
            case [float() | int(), float() | int()] if len(self.axes) == 1: #type: ignore
                self.zlimits = [args]
            case [tuple(), *_] if len(self.axes) == len(args): #type: ignore
                self.zlimits = args
            case _:
                raise ValueError(f'Invalid zlimits: {args}')

    def set_zlimit(self, *args):
        self.set_zlimits(*args)

    def clear(self, *args, **kwargs):
        for ax in self.axes: #type: ignore
            ax.clear()

    def draw_titles(self):
        if self.titles is not None:
            for ax, title in zip(self.axes, self.titles): #type: ignore
                ax.set_title(title)

    def draw_limits(self):
        if self.xlimits is not None:
            for ax, xlimits in zip(self.axes, self.xlimits): #type: ignore
                ax.set_xlim(*xlimits)
        if self.ylimits is not None:
            for ax, ylimits in zip(self.axes, self.ylimits): #type: ignore
                ax.set_ylim(*ylimits)
        if self.zlimits is not None:
            for ax, zlimits in zip(self.axes, self.zlimits): #type: ignore
                ax.set_zlim(*zlimits)

class NeuronFigure2D(NeuronFigure):
    def __init__(self, plot_gates: bool = True, figsize: tuple[int, int] = (20, 10)):
        shape = (1, 2) if plot_gates else (1, 1)
        self.plot_gates = plot_gates
        self.fig, self.axes = plt.subplots(*shape, figsize=figsize)
        self.axes = self.axes.flatten() if plot_gates else np.array([self.axes])
        self.set_defaults()

    @property
    def figure(self):
        return self.fig

    def plot(self, V, m=None, n=None, h=None):
        self.clear()
        self.draw_titles()
        self.draw_limits()

        self.axes[0].plot(V)
        if self.plot_gates:
            self.axes[1].plot(m, label='m')
            self.axes[1].plot(n, label='n')
            self.axes[1].plot(h, label='h')
            self.axes[1].legend()


def dict_get(d: dict, *keys: Any, default: Any = None) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return default

def draw_path(
    num_points: int,
    dr:         float,
    chance:     float = 0.5,
    min_theta:  float = -PI/4,
    max_theta:  float =  PI/4,
    rot_scale:  float = 0.2,
    penalty:    float = 0.1,
    **kwargs:   Any
):
    assert penalty > 0
    x = np.zeros(num_points)
    y = np.zeros(num_points)

    bounds = dict_get(
        kwargs, 'bounds', 'theta_bounds', 'theta_range', 
        default = None
    )
    match bounds:
        case None:
            pass
        case [lower, upper]:
            min_theta = lower
            max_theta = upper
        case int() | float():
            min_theta = -bounds
            max_theta =  bounds
        case _:
            raise ValueError(f'Invalid _bounds: {bounds}')

    theta = 0.

    x[0] = dr
    y[0] = 0.

    for i in range(1, num_points):
        if npr.random_sample() < chance:
            theta += (npr.random_sample()-0.5)*rot_scale

        if theta <= min_theta:
            theta += 0.01
        elif theta >= max_theta:
            theta -= 0.01

        dx = dr * np.cos(theta)
        dy = dr * np.sin(theta)
        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dy

    return x, y

class NeuronFigure3D(NeuronFigure):
    def __init__(self, figsize: tuple[int, int] = (20, 10), **kwargs):
        self.fig     = plt.figure(figsize=figsize)
        self.axes    = np.array([self.fig.add_subplot(projection='3d', **kwargs)])
        self.pathx   = None
        self.pathy   = None
        self.pathz   = None
        self.path_color = 'blue'

        self.set_defaults()

    @property
    def figure(self):
        return self.fig

    def set_path(self, pathx: np.ndarray, pathy: np.ndarray):
        self.pathx = pathx
        self.pathy = pathy
        self.pathz = np.zeros_like(pathx)

    def plot_path(self):
        self.axes[0].plot(
            self.pathx,
            self.pathy,
            self.pathz,
            color=self.path_color
        )

    def plot(self, V, *_):
        self.clear()
        self.draw_titles()
        self.draw_limits()
        self.plot_path()

        self.axes[0].plot(self.pathx, self.pathy, V)

class AP:

    NilMode       = 0
    IterMode      = 1
    EnumerateMode = 2

    N: int
    dx: float

    @staticmethod
    def set_params(**kwargs):
        AP.N, AP.dx = neuro.set_params(**kwargs)
        return AP.N, AP.dx

    def __init__(self, dx: float, **kwargs):
        self.I : int | float | Callable = 0.

        dt_factor    = kwargs.pop('dt_factor', 0.2)
        dt_init      = kwargs.pop('dt_init', None)

        _, dx = self.set_params(dx=dx, **kwargs)

        self.mode = AP.NilMode

        self.count = 0
        self.t = 0.
        self.V : np.ndarray = np.zeros(AP.N)
        self.m : np.ndarray = np.zeros(AP.N)
        self.n : np.ndarray = np.zeros(AP.N)
        self.h : np.ndarray = np.zeros(AP.N)

        self.dt_init = dt_init if dt_init is not None else dt_factor*dx*dx
        self.dt      = self.dt_init
        
        self.set_iter_params()

    def set_initial_voltage(self, *args, **kwargs):
        _V0 = None
        match args, kwargs:
            case [float(V0) | int(V0)], {**extra} if not extra:
                _V0 = np.zeros(AP.N) + V0
            case [np.ndarray(V0)], {**extra} if not extra:
                assert V0.shape == (AP.N,)
                _V0 = V0
            case (
                ([float(V0) | int(V0), (lo, hi)], _)  | 
                (
                    [float(V0) | int(V0), *_], 
                    {'lo': lo, 'hi': hi} | 
                    {'start': lo, 'end': hi}
                )
            ):
                if isinstance(lo, float) and isinstance(hi, float):
                    assert 0. <= lo < hi <= 1.
                    lo = int(round(lo*AP.N))
                    hi = int(round(hi*AP.N))
                _V0 = np.zeros(AP.N)
                _V0[lo:hi] = V0
            case _:
                raise ValueError(f'Invalid args: {args}, {kwargs}')

        self.set_initial_conditions(V0=_V0)


    def set_initial_conditions (
        self,
        V0: int | float | np.ndarray        = 0.0001,
        m0: int | float | np.ndarray | None = None,
        n0: int | float | np.ndarray | None = None,
        h0: int | float | np.ndarray | None = None,
    ):
        if isinstance(V0, (int, float)):
            V0 = np.ones(AP.N)*V0

        if m0 is None:
            m0 = np.ones(AP.N)*(
                a_m(V0)/(a_m(V0)+b_m(V0))
            )
        elif isinstance(m0, (int, float)):
            m0 = np.ones(AP.N)*m0

        if n0 is None:
            n0 = np.ones(AP.N)*(a_n(V0)/(a_n(V0)+b_n(V0)))
        elif isinstance(n0, (int, float)):
            n0 = np.ones(AP.N)*n0

        if h0 is None:
            h0 = np.ones(AP.N)*(a_h(V0)/(a_h(V0)+b_h(V0)))
        elif isinstance(h0, (int, float)):
            h0 = np.ones(AP.N)*h0

        self.V0, self.m0, self.n0, self.h0 = V0, m0, n0, h0

    def set_current(self, I: int | float | Callable, max_time: Optional[float] = None):
        if max_time is not None:
            assert not callable(I)
            self.I = lambda t: I if t < max_time else 0
        else:
            self.I = I


    def set_iter_params(
        self,
        tolerance:   float = TOLERANCE,
        interval:    float = 0.01,
        T:           float = 2,
        verbose:     bool  = True,
        get_dt_list: bool = False
    ):
        self.tolerance = tolerance
        self.interval  = interval
        self.T         = T
        self.verbose   = verbose
        self.get_dt_list = get_dt_list

    def _iter_prepare(self):
        self.count = 0
        self.t = 0.

        if any(e is None for e in (self.V0, self.m0, self.n0, self.h0)):
            raise ValueError('Call set_initial_conditions first')
        self.V = self.V0.copy() #type: ignore
        self.m = self.m0.copy() #type: ignore
        self.n = self.n0.copy() #type: ignore
        self.h = self.h0.copy() #type: ignore
        self.dt = self.dt_init

    def __iter__(self):
        if self.mode != AP.NilMode:
            return self

        self._iter_prepare()
        self.mode = AP.IterMode

        return self

    def enumerate(self):
        self._iter_prepare()
        self.mode = AP.EnumerateMode

        return self

    def __next__(self) -> Any:
        if self.mode == AP.NilMode:
            raise ValueError('Call __iter__ first')
        elif self.t == 0.:
            self.t += self.dt
            if self.mode == AP.IterMode:
                return self.V, self.m, self.n, self.h
            elif self.mode == AP.EnumerateMode:
                return 0., (self.V, self.m, self.n, self.h)

        self.dt, *extra = neuro.get_next_n(
            self.V, self.m, self.n, self.h,
            self.tolerance, self.I, self.t,
            self.dt, self.interval, self.get_dt_list
        )
        dist = extra[0]
        dt_list = extra[1] if self.get_dt_list else None
        if self.verbose:
            print(f'[#{self.count}] {self.t:.3f} -> {self.T} '
                  f'{{dt={self.dt:.4e}}}'+' '*10, end='\r')
        self.t += dist
        return_tuple = None

        if self.get_dt_list:
            return_tuple = self.V, self.m, self.n, self.h, dt_list
        else:
            return_tuple = self.V, self.m, self.n, self.h
        if self.t >= self.T:
            self.mode = AP.NilMode
            raise StopIteration
        self.count += 1
        if self.mode == AP.IterMode:
            return return_tuple
        elif self.mode == AP.EnumerateMode:
            return self.t, (return_tuple)



# ------------------------------------------------


def animate(
        save_name: str,
        figure: NeuronFigure,
        neuron: AP,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> None:

    output_path = None
    if save_path is not None:
        output_path = path.realpath(
            path.join(path.expanduser(save_path), path.expanduser(save_name))
        )
    else:
        output_path = path.realpath(path.expanduser(save_name))

    writer = mpl_anim.FFMpegWriter(fps=30)

    print(f'Saving to {output_path}\n' + '-'*30 + '\n')

    with writer.saving(figure.figure, output_path, dpi=dpi):
        t_index = 0
        for e in neuron:
            figure.plot(*e)
            writer.grab_frame()
            t_index += 1


if __name__ == '__main__':

    T = 1.6
    draw_interval = 0.01
    base_args = dict (
        dx=0.005,
        D=0.5,
        E_K=-12.0,
        dt_factor=0.5,
        verbosity=1
    )

    "Create the neuron object"
    neuron = AP(**base_args)
    neuron.set_initial_voltage(0.)
    # neuron.set_initial_voltage(10.05, start=0.8, end=1.0)
    # neuron.set_current(1.2)
    # neuron.set_current( lambda t: np.max([1.4*np.cos(2.*t), 0.]) )
    neuron.set_current(0.2, max_time=0.1)
    # neuron.set_current(
    #         lambda t: 1.0 if int(round(1.5*t)) %2 else 0
    #         )
    # neuron.set_current(
            # lambda t: 1.0 if int(round(1.5*t)) %2 else 0
            # )
    # neuron.set_current(1.2)
    neuron.set_iter_params(interval=draw_interval, T=T, tolerance=5e-4)

    "Create the figure object"
    # fig = NeuronFigure2D(True, figsize=(20,10))
    # fig.set_ylimits((-60, 250), (-0.2, 1.2))
    # fig = NeuronFigure2D(plot_gates=False, figsize=(10,10))
    # fig.set_title(ax_title, '')
    # fig.set_ylimits(-60, 250)

    "(could also be a 3d figure)"
    X, Y = draw_path(neuron.N, neuron.dx, chance=0.75, bounds=PI/3, rot_scale=0.4)
    fig = NeuronFigure3D(figsize=(10,10), elev=20, azim=-85)
    # # fig.set_title(ax_title)
    fig.set_zlimit(-60, 250)
    fig.set_path(X, Y) 


    animate('cool__test.mp4', fig, neuron, save_path='~/Desktop', dpi=150)



