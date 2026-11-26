"""Tests for the beamsplitter and the forked optical system."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TEST_FRAME
from hwoutils.numerics import same_tolerance
from hwoutils.testing import ulps
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Band, Quadrature

from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import _grid_axis
from physicaloptix.elements import SampledOptic, SampledPhaseScreen
from physicaloptix.instruments.nircam import NIRCamBand, NIRCamInputs
from physicaloptix.instruments.roman import RomanCompact
from physicaloptix.instruments.roman_full import RomanFull
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import PointSource, broadcast_to_spectrum
from physicaloptix.system import BeamSplitter, Branch, OpticalSystem
from physicaloptix.transforms import Fraunhofer

NPIX = 16


def _pupil_field(npix=NPIX):
    grid = PixelGrid.pupil(npix)
    x = np.asarray(_grid_axis(grid))
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.25).astype(complex)
    ramp = np.exp(1j * 2 * np.pi * 1.5 * xx)  # structure, so ports differ
    field = Field(data=jnp.asarray(disk * ramp), grid=grid, plane=PlaneKind.PUPIL)
    return field, grid


def _binary_mask(npix=NPIX, radius=0.2):
    x = np.asarray(_grid_axis(PixelGrid.pupil(npix)))
    xx, yy = np.meshgrid(x, x)
    return ((xx**2 + yy**2) <= radius**2).astype(float)


class TestFromMask:
    def test_transmit_port_bit_identical_to_sampled_optic(self):
        field, grid = _pupil_field()
        mask = _binary_mask()
        split = BeamSplitter.from_mask(
            jnp.asarray(mask), grid=grid, plane=PlaneKind.PUPIL
        )
        optic = SampledOptic(
            transmission=jnp.asarray(mask), grid=grid, plane=PlaneKind.PUPIL
        )
        ports = split(field)
        np.testing.assert_array_equal(
            np.asarray(ports["transmit"].data), np.asarray(optic(field).data)
        )

    def test_binary_babinet_conserves_energy_exactly(self):
        field, grid = _pupil_field()
        mask = _binary_mask()
        split = BeamSplitter.from_mask(
            jnp.asarray(mask), grid=grid, plane=PlaneKind.PUPIL
        )
        assert float(split.energy_residual) == 0.0
        ports = split(field)
        total = (
            np.abs(np.asarray(ports["transmit"].data)) ** 2
            + np.abs(np.asarray(ports["reflect"].data)) ** 2
        )
        np.testing.assert_allclose(
            total, np.abs(np.asarray(field.data)) ** 2, atol=1e-15
        )

    def test_reject_port_is_the_babinet_complement(self):
        field, grid = _pupil_field()
        mask = _binary_mask()
        split = BeamSplitter.from_mask(
            jnp.asarray(mask), grid=grid, plane=PlaneKind.PUPIL
        )
        ports = split(field)
        np.testing.assert_array_equal(
            np.asarray(ports["reflect"].data),
            np.asarray(field.data) * (1.0 - mask),
        )

    def test_refuses_grey_mask(self):
        _, grid = _pupil_field()
        gray = jnp.full((NPIX, NPIX), 0.5)
        with pytest.raises(ValueError, match="binary"):
            BeamSplitter.from_mask(gray, grid=grid, plane=PlaneKind.PUPIL)


class TestEnergySplit:
    @pytest.mark.parametrize("reflectance", [0.2, 0.5])
    def test_lossless_split_at_float32_precision(self, reflectance):
        """Accept float32 energy roundoff when rebuilding the split under x64."""
        grid = PixelGrid.pupil(2)
        with jax.enable_x64(False):
            split = BeamSplitter.energy(reflectance, grid=grid, plane=PlaneKind.PUPIL)
            assert split.t.dtype == jnp.complex64
            assert split.energy_residual <= np.finfo(np.float32).eps

        with jax.enable_x64(True):
            widened = BeamSplitter(
                t=split.t, r=split.r, grid=grid, plane=PlaneKind.PUPIL
            )
            assert widened.energy_residual <= np.finfo(np.float32).eps

    def test_conserves_and_uses_quadrature_phase(self):
        """Floating-point: two roots, complex products, norms squared and a sum
        cost at most 16 roundings of incident intensity. Division of the
        complex ports costs eight roundings at the modulus of their ratio.
        """
        field, grid = _pupil_field()
        split = BeamSplitter.energy(0.3, grid=grid, plane=PlaneKind.PUPIL)
        assert float(split.energy_residual) <= ulps(4, split.t.real.dtype.type(1))
        ports = split(field)
        total = (
            np.abs(np.asarray(ports["transmit"].data)) ** 2
            + np.abs(np.asarray(ports["reflect"].data)) ** 2
        )
        np.testing.assert_allclose(
            total,
            np.abs(np.asarray(field.data)) ** 2,
            rtol=0,
            atol=ulps(16, np.max(np.abs(field.data)) ** 2),
        )
        # The lossless-symmetric convention puts the ports in quadrature.
        transmit = np.asarray(ports["transmit"].data)
        reflect = np.asarray(ports["reflect"].data)
        lit = transmit != 0
        ratio = reflect[lit] / transmit[lit]
        np.testing.assert_allclose(
            np.real(ratio), 0.0, rtol=0, atol=ulps(8, np.max(np.abs(ratio)))
        )

    @pytest.mark.parametrize("direction", [-1, 0, 1])
    def test_energy_gate_uses_original_amplitudes(self, direction):
        """Host amplitudes straddling the energy limit can narrow to one value.

        Args:
            direction: Select the lossless point or either side of the limit.
        """
        grid = PixelGrid.pupil(2)
        threshold = np.sqrt(same_tolerance(1.0, ulps=4))
        r = 0.0 if direction == 0 else threshold * (1 + direction * 2.0**-26)

        def build():
            return BeamSplitter(t=1.0, r=r, grid=grid, plane=PlaneKind.PUPIL).r

        for call in (build, jax.jit(build)):
            if direction > 0:
                with pytest.raises(ValueError, match="energy residual"):
                    call()
            else:
                actual = call()
                assert float(actual) == np.asarray(r, dtype=actual.dtype)

    def test_energy_gate_raises_by_default(self):
        _, grid = _pupil_field()
        with pytest.raises(ValueError, match="energy"):
            BeamSplitter(
                t=jnp.asarray(0.9),
                r=jnp.asarray(0.1),
                grid=grid,
                plane=PlaneKind.PUPIL,
            )

    def test_energy_gate_record_policy_builds_and_records(self):
        _, grid = _pupil_field()
        split = BeamSplitter(
            t=jnp.asarray(0.9),
            r=jnp.asarray(0.1),
            grid=grid,
            plane=PlaneKind.PUPIL,
            on_violation="record",
        )
        assert float(split.energy_residual) > 0.1

    def test_refuses_bare_nlam_transmission(self):
        _, grid = _pupil_field()
        with pytest.raises(ValueError, match="rank"):
            BeamSplitter(
                t=jnp.ones(3) / np.sqrt(2.0),
                r=jnp.ones(3) / np.sqrt(2.0),
                grid=grid,
                plane=PlaneKind.PUPIL,
            )


class TestDichroic:
    def test_routes_color_and_conserves_per_wavelength(self):
        """Floating-point: interpolation, two roots, port products, two
        squared norms and their sum cost 20 roundings at incident intensity.
        """
        mono, grid = _pupil_field()
        # Step from transmit to reflect across the band.
        split = BeamSplitter.dichroic(
            wavelengths_nm=jnp.asarray([440.0, 560.0]),
            transmittance=jnp.asarray([1.0, 0.0]),
            grid=grid,
            plane=PlaneKind.PUPIL,
        )
        band = Band(430.0, 570.0)
        system, _, _ = _two_arm_system()
        system = eqx.tree_at(lambda s: s.split, system, split)
        channel, _ = system.as_channel_path("sci")
        np.testing.assert_array_equal(channel.spectral_breaks_nm(), [440, 560])
        spectrum = Quadrature.build((band,), system, order=1)
        field = broadcast_to_spectrum(mono, spectrum)
        np.testing.assert_array_equal(spectrum.nodes_nm[0], [435.0, 500.0, 565.0])
        ports = split(field)
        t_int = np.abs(np.asarray(ports["transmit"].data)) ** 2
        r_int = np.abs(np.asarray(ports["reflect"].data)) ** 2
        np.testing.assert_allclose(
            t_int + r_int,
            np.abs(np.asarray(field.data)) ** 2,
            rtol=0,
            atol=ulps(20, np.max(np.abs(field.data)) ** 2),
        )
        # The blue end mostly transmits; the red end mostly reflects.
        assert t_int[0].sum() > r_int[0].sum()
        assert r_int[-1].sum() > t_int[-1].sum()

    def test_requires_a_chromatic_field(self):
        mono, grid = _pupil_field()
        split = BeamSplitter.dichroic(
            wavelengths_nm=jnp.asarray([440.0, 560.0]),
            transmittance=jnp.asarray([1.0, 0.0]),
            grid=grid,
            plane=PlaneKind.PUPIL,
        )
        with pytest.raises(ValueError, match="spectrum"):
            split(mono)


class _Counting(eqx.Module):
    """A pupil optic that counts its applications (eager-mode test helper)."""

    inner: SampledOptic
    calls: list = eqx.field(static=True)

    @property
    def plane(self):
        return self.inner.plane

    def __call__(self, field):
        self.calls.append(1)
        return self.inner(field)


def _two_arm_system(npix=NPIX, counting=False):
    field, grid = _pupil_field(npix)
    focal = PixelGrid.focal(2 * npix, 0.5, TEST_FRAME)
    disk = _binary_mask(npix, radius=0.45)
    stop = SampledOptic(
        transmission=jnp.asarray(disk), grid=grid, plane=PlaneKind.PUPIL
    )
    calls = []
    op = _Counting(inner=stop, calls=calls) if counting else stop
    trunk = OpticalPath(stages=(Stage("stop", op),))
    split = BeamSplitter.energy(0.2, grid=grid, plane=PlaneKind.PUPIL)
    sci = OpticalPath(
        stages=(Stage("science", Fraunhofer(grid_in=grid, grid_out=focal)),)
    )
    wfs = OpticalPath(stages=())  # a bare pupil (re-imaging) detector arm
    system = OpticalSystem(
        trunk=trunk,
        split=split,
        branches=(Branch("sci", "transmit", sci), Branch("wfs", "reflect", wfs)),
    )
    return system, field, calls


class TestOpticalSystem:
    def test_propagates_every_branch(self):
        system, field, _ = _two_arm_system()
        outs, taps = system.propagate(field)
        assert set(outs) == {"sci", "wfs"}
        assert outs["sci"].plane is PlaneKind.FOCAL
        assert outs["wfs"].plane is PlaneKind.PUPIL
        assert taps == {}

    def test_trunk_runs_exactly_once(self):
        system, field, calls = _two_arm_system(counting=True)
        system.propagate(field)
        assert len(calls) == 1  # one trunk pass feeds both branches

    def test_matches_manual_composition(self):
        system, field, _ = _two_arm_system()
        outs, _ = system.propagate(field)
        trunk_out, _ = system.trunk.propagate(field)
        ports = system.split(trunk_out)
        sci_manual, _ = system.branches[0].path.propagate(ports["transmit"])
        np.testing.assert_array_equal(
            np.asarray(outs["sci"].data), np.asarray(sci_manual.data)
        )
        np.testing.assert_array_equal(
            np.asarray(outs["wfs"].data), np.asarray(ports["reflect"].data)
        )

    def test_namespaced_taps(self):
        system, field, _ = _two_arm_system()
        _, taps = system.propagate(field, taps=("trunk/stop", "sci/science"))
        assert set(taps) == {"trunk/stop", "sci/science"}
        assert taps["trunk/stop"].plane is PlaneKind.PUPIL
        assert taps["sci/science"].plane is PlaneKind.FOCAL

    def test_unknown_tap_raises(self):
        system, field, _ = _two_arm_system()
        with pytest.raises(ValueError, match="tap"):
            system.propagate(field, taps=("sci/nonexistent",))
        with pytest.raises(ValueError, match="tap"):
            system.propagate(field, taps=("nobranch/stop",))

    def test_chromatic_threads_through(self):
        system, mono, _ = _two_arm_system()
        spectrum = Quadrature.build(
            (Band(500.0 * (1 - 0.1 / 2), 500.0 * (1 + 0.1 / 2)),), order=3
        )
        field = broadcast_to_spectrum(mono, spectrum)
        outs, _ = system.propagate(field)
        assert outs["sci"].data.shape[0] == 3
        assert outs["wfs"].spectrum is spectrum

    def test_rejects_plane_discontinuity_at_construction(self):
        system, _, _ = _two_arm_system()
        focal = PixelGrid.focal(2 * NPIX, 0.5, TEST_FRAME)
        bad_first = SampledOptic(
            transmission=jnp.ones((2 * NPIX, 2 * NPIX)),
            grid=focal,
            plane=PlaneKind.FOCAL,
        )
        with pytest.raises(ValueError, match="plane"):
            OpticalSystem(
                trunk=system.trunk,
                split=system.split,
                branches=(
                    Branch(
                        "bad", "transmit", OpticalPath(stages=(Stage("m", bad_first),))
                    ),
                ),
            )

    def test_rejects_duplicate_branch_names(self):
        system, _, _ = _two_arm_system()
        with pytest.raises(ValueError, match="duplicate"):
            OpticalSystem(
                trunk=system.trunk,
                split=system.split,
                branches=(system.branches[0], system.branches[0]),
            )

    def test_rejects_unknown_port(self):
        system, _, _ = _two_arm_system()
        with pytest.raises(ValueError, match="port"):
            OpticalSystem(
                trunk=system.trunk,
                split=system.split,
                branches=(Branch("sci", "backdoor", system.branches[0].path),),
            )


class TestAsChannelPath:
    def test_flattened_channel_matches_container(self):
        system, field, _ = _two_arm_system()
        outs, _ = system.propagate(field)
        for name in ("sci", "wfs"):
            flat, _ = system.as_channel_path(name)
            out_flat, _ = flat.propagate(field)
            np.testing.assert_allclose(
                np.asarray(out_flat.data), np.asarray(outs[name].data), atol=1e-15
            )

    def test_flattened_chromatic_matches(self):
        system, mono, _ = _two_arm_system()
        field = broadcast_to_spectrum(
            mono,
            Quadrature.build(
                (Band(500.0 * (1 - 0.1 / 2), 500.0 * (1 + 0.1 / 2)),), order=3
            ),
        )
        outs, _ = system.propagate(field)
        flat, _ = system.as_channel_path("sci")
        out_flat, _ = flat.propagate(field)
        np.testing.assert_allclose(
            np.asarray(out_flat.data), np.asarray(outs["sci"].data), atol=1e-15
        )

    def test_index_map_locates_every_stage(self):
        system, _, _ = _two_arm_system()
        flat, index_map = system.as_channel_path("sci")
        assert index_map["stop"] == 0  # trunk stage first
        assert index_map["sci_port"] == 1  # the port shim
        assert index_map["science"] == 2  # branch stage after
        assert flat.stages[index_map["science"]].name == "science"

    def test_unknown_branch_raises(self):
        system, _, _ = _two_arm_system()
        with pytest.raises(ValueError, match="branch"):
            system.as_channel_path("nope")


class TestFoldIntegration:
    def test_cannot_be_a_stage_of_the_linear_fold(self):
        _, grid = _pupil_field()
        split = BeamSplitter.energy(0.5, grid=grid, plane=PlaneKind.PUPIL)
        with pytest.raises(ValueError, match="multi-output"):
            OpticalPath(stages=(Stage("split", split),))

    def test_validates_plane_and_grid(self):
        field, grid = _pupil_field()
        split = BeamSplitter.energy(0.5, grid=grid, plane=PlaneKind.PUPIL)
        bad_plane = Field(data=field.data, grid=grid, plane=PlaneKind.FOCAL)
        with pytest.raises(ValueError, match="plane"):
            split(bad_plane)


class TestOptixstuffSeam:
    """A flattened science branch serves the optixstuff coronagraph seam."""

    def test_science_branch_wraps_into_a_path_coronagraph(self):
        """A flattened science branch supplies the public coronagraph interface.

        The vortex throughput curve must have a positive, finite half-maximum
        crossing. This checks the existence of the crossing and finite samples,
        so one sky screen suffices and no numerical tolerance is needed.
        """
        from optixstuff.coronagraph import AbstractCoronagraph

        from physicaloptix.elements import MultiScaleVortex
        from physicaloptix.interop import PathCoronagraph

        npup = 32
        grid = PixelGrid.pupil(npup)
        x = np.asarray(_grid_axis(grid))
        xx, yy = np.meshgrid(x, x)
        disk = ((xx**2 + yy**2) <= 0.25).astype(float)
        lyot = ((xx**2 + yy**2) <= 0.45**2).astype(float)
        trunk = OpticalPath(
            stages=(
                Stage(
                    "stop",
                    SampledOptic(
                        transmission=jnp.asarray(disk),
                        grid=grid,
                        plane=PlaneKind.PUPIL,
                    ),
                ),
            )
        )
        split = BeamSplitter.energy(0.1, grid=grid, plane=PlaneKind.PUPIL)
        sci = OpticalPath(
            stages=(
                Stage(
                    "vortex",
                    MultiScaleVortex.build(
                        charge=2, npup=npup, q=32, scaling_factor=4, window_size=8
                    ),
                ),
                Stage(
                    "lyot",
                    SampledOptic(
                        transmission=jnp.asarray(lyot),
                        grid=grid,
                        plane=PlaneKind.PUPIL,
                    ),
                ),
            )
        )
        system = OpticalSystem(
            trunk=trunk,
            split=split,
            branches=(
                Branch("sci", "transmit", sci),
                Branch("lowfs", "reflect", OpticalPath(stages=())),
            ),
        )
        field = Field(
            data=jnp.asarray(disk).astype(complex), grid=grid, plane=PlaneKind.PUPIL
        )
        flat, _ = system.as_channel_path("sci")
        from optixstuff import SimplePrimary

        coronagraph = PathCoronagraph.from_path(
            flat,
            field,
            primary=SimplePrimary(
                circumscribed_diameter_m=6.0, collecting_area_m2=9 * np.pi
            ),
            reference_wavelength_nm=600.0,
            max_separation_lod=6.0,
            pixel_scale_lod=0.5,
            n_sky_screens=1,
        )
        assert isinstance(coronagraph, AbstractCoronagraph)
        assert (
            np.isfinite(float(coronagraph.curves().iwa_lod))
            and float(coronagraph.curves().iwa_lod) > 0
        )
        assert np.all(np.isfinite(np.asarray(coronagraph.curves().throughput)))


def _roman_compact(fpm, cast):
    flat = cast(np.zeros((4, 4)))
    return RomanCompact(
        pupil=flat,
        lyot=flat,
        pupil_mask=flat,
        fpm=cast(fpm),
        kind="hlc",
        n_small=4,
        n_big=4,
        pupil_diam_pix=2.0,
        fpm_sampling_lod=1.0,
        fpm_lam0_nm=500.0,
        lam0_nm=500.0,
        dm1_surface_m=flat,
        dm2_surface_m=flat,
        dm1_wfe_m=flat,
        dm2_wfe_m=flat,
    )


def _leaf_cases(cast):
    grid = PixelGrid.pupil(NPIX)
    flat = cast(np.ones((NPIX, NPIX)))
    return {
        "sampled_optic": (
            SampledOptic(flat, grid, PlaneKind.PUPIL, shift_px=cast(np.zeros(2))),
            ["transmission", "shift_px"],
            "float",
        ),
        "sampled_optic_phasor": (
            SampledOptic(cast(np.ones((NPIX, NPIX)) * 1j), grid, PlaneKind.PUPIL),
            ["transmission"],
            "complex",
        ),
        "phase_screen": (SampledPhaseScreen(flat, grid), ["opd_nm"], "float"),
        "point_source": (
            PointSource(
                cast([0.1, -0.2]), flux=cast(2.0), spectral_weights=cast([1.0, 0.5])
            ),
            ["position_arcsec", "flux", "spectral_weights"],
            "float",
        ),
        "beam_splitter": (
            BeamSplitter(t=cast(1.0), r=cast(0.0), grid=grid, plane=PlaneKind.PUPIL),
            ["t", "r"],
            "float",
        ),
        "dichroic": (
            BeamSplitter(
                wavelengths_nm=cast([500.0, 600.0]),
                transmit_table=cast([0.5, 0.25]),
                grid=grid,
                plane=PlaneKind.PUPIL,
            ),
            ["wavelengths_nm", "transmit_table"],
            "float",
        ),
        "roman_compact": (
            _roman_compact(np.ones((3, 3)), cast),
            [
                "pupil",
                "lyot",
                "pupil_mask",
                "fpm",
                "dm1_surface_m",
                "dm2_surface_m",
                "dm1_wfe_m",
                "dm2_wfe_m",
            ],
            "float",
        ),
        "roman_compact_complex_mask": (
            _roman_compact(np.ones((3, 3), complex), cast),
            ["fpm"],
            "complex",
        ),
        "nircam_inputs": (
            NIRCamInputs(
                pupil_amplitude=flat,
                pupil_diameter_m=6.5,
                ote_opd_nm=flat,
                lyot_amplitude=flat,
                si_opd_nm=flat,
                si_support=flat,
                si_wavelength_nm=3300.0,
            ),
            [
                "pupil_amplitude",
                "ote_opd_nm",
                "lyot_amplitude",
                "si_opd_nm",
                "si_support",
            ],
            "float",
        ),
        "nircam_band": (
            NIRCamBand(
                wavelengths_nm=(3300.0,),
                weights=(1.0,),
                si_opd_nm=cast(np.zeros((1, 4, 4))),
            ),
            ["si_opd_nm"],
            "float",
        ),
    }


def _narrow(value):
    """NumPy float32 or complex64, the input a float64 build must widen."""
    return np.asarray(value, np.complex64 if np.iscomplexobj(value) else np.float32)


def _wide(value):
    """NumPy float64 or complex128, the input a float32 build must narrow."""
    return np.asarray(value, complex if np.iscomplexobj(value) else float)


@pytest.mark.parametrize("case", list(_leaf_cases(_wide)))
def test_converted_fields_are_strong_leaves(case, precision):
    """NumPy arrays and scalars given at construction are stored as strongly
    typed arrays of the active dtype whatever their own width: float32 (a
    complex value complex64) under x32 from float64 input, float64 (complex128)
    under x64 from float32 input. Truth source: the leaf rule; dtypes are
    exact, so there is no tolerance."""
    x64 = precision == np.dtype("float64")
    module, names, kind = _leaf_cases(_narrow if x64 else _wide)[case]
    expected = {
        "float": jnp.float64 if x64 else jnp.float32,
        "complex": jnp.complex128 if x64 else jnp.complex64,
    }[kind]
    for name in names:
        leaf = getattr(module, name)
        assert isinstance(leaf, jax.Array), name
        assert leaf.dtype == expected, name
        assert not leaf.weak_type, name


@pytest.mark.float32_only(
    reason="checks narrowing of host floats and phasors to float32 and complex64"
)
def test_roman_full_arrays_are_strong_leaves_of_their_kind_under_x32():
    """Each array keeps its kind (a phasor stays complex, a mask stays boolean)
    in the active precision. Truth source: the leaf rule; exact dtypes."""
    model = RomanFull(
        arrays={
            "tilt": np.ones((2, 2), complex),
            "map": np.zeros((2, 2)),
            "mask": np.ones((2, 2), bool),
        },
        program=(),
        n=2,
        lam_m=5e-7,
        pupil_diam_pix=1.0,
        final_dx_m=1e-5,
    )
    want = {"tilt": jnp.complex64, "map": jnp.float32, "mask": jnp.bool_}
    for key, dtype in want.items():
        assert model.arrays[key].dtype == dtype, key
        assert not model.arrays[key].weak_type, key
