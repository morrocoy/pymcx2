# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Mon Mar 22 17:58:19 2021

@author: kpapke
"""
import sys
import os.path
import subprocess
import re
from ast import literal_eval

import numpy as np
import json

from .bindings import ordered_dict
from .findmcx import findMCX
from .mc2store import loadmc2
from .mchstore import loadmch

from .log import logmanager
# from tempfile import NamedTemporaryFile

__all__ = ['MCSession']

logger = logmanager.getLogger(__name__)


class MCSession(object):
    """ Represents a mcx simulation session.

    """

    def __init__(self, name, workdir, seed=-1, autoload=False):
        """ Constructor.

        Parameters
        ----------
        name : str
            The session name or id used as default name for the output files.
        workdir : str or pathlib.Path,
            The path to the working directory.
        seed : int
            The seed of the CPU random number generator (RNG). A value of -1
            let MCX to automatically seed the CPU-RNG using system clock. A
            value n > 0 sets the CPU-RNG's seed to n.
        """
        self.name = name  # seesion id name
        self.workdir = workdir  # working directory
        self.seed = seed  # seed of the CPU random number generator (automatic)

        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        # boundary settings
        self.boundary = ordered_dict({
            'specular': True,  # enable specular reflection
            'missmatch': True,  # considers refractive index mismatch
            'n0': 1,  # background refractive index
            'bc': "______000000",  # per-face boundary conditions.
        })

        # detectors settings
        self.detector = ordered_dict({
        })

        # domain settings
        self.domain = ordered_dict({
            'vol': None,  # domain volume as 2-d or 3-d array
            'shape': tuple(),  # shape of vol
            'origin': 0,  # coordinate origin mode of the volume [1 1 1].
            'scale': 1.,  # edge length in mm
        })

        # time domain setting
        self.forward = ordered_dict({
            't0' : 0.0,  # start time
            't1': 5.0e-9,  # end time
            'dt': 5.0e-9,  # time step
            'ntime': 1,  # number of time steps
        })

        # material settings (initialize with background material of tag 0)
        self.material = ordered_dict({
            'mat0': ordered_dict({
                'tag': 0,
                'mua': 0.,
                'mus': 0.,
                'g': 1.,
                'n': 1.
            })
        })


        # output settings
        self.output = ordered_dict({
            'type': "X",  # type of data to be saved in the volumetric output
            'normalize': True,  # normalize the solutions
            'mask': "DP",  # parameters to be saved for detected photons.
            'format': "mc2",  # volumetric data output format
        })

        # source settings
        self.source = ordered_dict({
            'nphoton': 1e3,  # total number of photons to be simulated
            'pos': [0, 0, 0],  # grid position of a source in grid unit
            'dir': [0, 0, 1],  # directional vector of the photon at launch
            'type': 'pencil',  # source type (pencil, isotropic, cone, ...)
            'param1': None,  # source type parameters
            'param2': None,  # additional Source type parameters
            'pattern': None  # additional Source type parameters
        })

        # result stores
        self.fluence = None  # four-dimensional numpy array (nx, ny, nz, ntime)
        self.detectedPhotons = None  # pandas dataframe

        # simulation statistics
        self.stat = ordered_dict({
            'normalizer': None,  # normalization factor
            'nphoton': None,  # total simulated photon number
            'nthread': None,  # total number of threads
            'runtime': None,  # simulation run-time per photon [ms/photon]
            'energytot': None,  # total initial weight/energy of all launched photons
            'energyabs': None,  # total absorbed weight/energy of all photons
        })

        filePath = ordered_dict({
            'config': os.path.join(self.workdir, self.name + ".json"),
            'volume': os.path.join(self.workdir, self.name + ".mcv"),
            'fluenc': os.path.join(self.workdir, self.name + ".mc2"),
            'detect': os.path.join(self.workdir, self.name + ".mch"),
        })

        if autoload:
            self.loadJSON(filePath['config'])  # retrieve configuration data
            self.loadVolume(filePath['volume'])  # retrieve volume data
            self.loadResults()  # retrieve fluence and detector data

        # else:
        #     for fp in filePath.values():
        #         if os.path.isfile(fp):
        #             os.remove(fp)


    def addDetector(self, pos, radius):
        """ Add a detector. The detector id is automatically chosen and
        returned.

        Parameters
        ----------
        id : int
            Identifiyer for detector.
        pos : float
            The grid position of a detector, in grid unit.
        radius : float
            The grid position of a detector, in grid unit.

        Returns
        -------
        id : int
            Identifiyer for the detector used for the simulation results.
        """
        id = len(self.detector)
        name = "det%d" % id
        self.detector[name] = ordered_dict({
            'id': id,
            'pos': pos,
            'radius': radius,
        })
        return id


    def addMaterial(self, mua, mus, g, n):
        """ Add a material with specified properties. The material tag is
        automatically chosen and returned.

        Parameters
        ----------
        mua : float
            Absortion coefficient [1/mm].
        mus : float
            Scattering coefficient [1/mm].
        g : float
            Anisothropy of scattering.
        n : float
            refractive index.

        Returns
        -------
        tag : int
            Identifiyer for material and associated with the the voxel value of
            the geometry domain.
        """
        tag = len(self.material)
        name = "mat%d" % tag
        self.material[name] = ordered_dict({
            'tag': tag,
            'mua': mua,
            'mus': mus,
            'g': g,
            'n': n
        })
        return tag


    def asJSON(self):
        """ Returns json object of the configuration.

        Returns
        -------
        cfg : json object
            The session configuration.
        """
        volumeFile = os.path.abspath(
            os.path.join(self.workdir, self.name+".mcv"))

        cfg = ordered_dict({
            'Session': ordered_dict({
                'ID': self.name,
                'RootPath': os.path.abspath(self.workdir),
                'Photons': int(self.source['nphoton']),
                'RNGSeed': self.seed,
                'DoMismatch': self.boundary['missmatch'],
                'DoSaveVolume': True,
                'DoNormalize': self.output['normalize'],
                'DoPartialPath': 'P' in self.output['mask'],
                'DoSaveRef': True,
                'DoDCS': 'M' in self.output['mask'],
                'DoSaveExit': 'X' in self.output['mask'],
                'DoSaveSeed': True,
                'DoSpecular': self.boundary['specular'],
                'DoAutoThread': True,
                'DebugFlag': 4,
                'SaveDataMask': self.output['mask'],
                'OutputFormat': self.output['format'],
                'OutputType': self.output['type']
            }),
            'Forward': ordered_dict({
                'T0': self.forward['t0'],
                'T1': self.forward['t1'],
                'Dt': self.forward['dt'],
                'N0': self.boundary['n0'],
            }),
            'Optode': ordered_dict({
                'Source': ordered_dict({
                    'Type': self.source['type'],
                    'Pos': self.source['pos'],
                    'Dir': self.source['dir'],
                }),
            }),
            'Domain': ordered_dict({
                'VolumeFile': volumeFile,
                'Dim': list(self.domain['vol'].shape),
                'OriginType': self.domain['origin'],
                'LengthUnit': self.domain['scale'],
                'Media': [
                    ordered_dict({
                        'mua': mat['mua'],
                        "mus": mat['mus'],
                        "g": mat['g'],
                        "n": mat['n']
                    }) for mat in self.material.values()
                ]
            }),
        })

        # optional source parameters
        for key in ('Param1', 'Param2', 'Pattern'):
            if self.source[key.lower()] is not None:
                cfg['Optode']['Source'][key] = self.source[key.lower()]

        # optional detector configuration
        if len(self.detector):
            cfg['Optode']['Detector'] = [
                ordered_dict({
                    'Pos': det['pos'],
                    'R': det['radius']
                })
                for det in self.detector.values()
            ]

        return json.dumps(cfg, sort_keys=False, indent=4)


    def dumpJSON(self, fileName=None):
        """ Export the configurarion in json format.

        Parameters
        ----------
        fileName : str, optional
            The filename. Default is the session id name.
        """
        if fileName is None:
            fileName = self.name + ".json"
        with open(os.path.join(self.workdir, fileName), 'w') as file:
            file.write(self.asJSON())


    def dumpVolume(self, fileName=None, order='F'):
        """ Export the geometry.

        Parameters
        ----------
        fileName : str, optional
            The filename. Default is the session id name.
        order : {'C', 'F', 'A'}, optional
            Controls the memory layout of the bytes object. 'C' means C-order,
            'F' means F-order, 'A' (short for Any) means 'F' if a is Fortran
            contiguous, 'C' otherwise. Default is 'F'.
        """
        if fileName is None:
            fileName = self.name + ".mcv"

        # if os.path.isfile(fp):
        #     os.remove(fp)

        with open(os.path.join(self.workdir, fileName), 'wb') as file:
            file.write(self.domain['vol'].tobytes(order=order))



    def clearFiles(self):
        """ Remove all simulation files. """
        for ext in ["json", "mch", "mc2", "mcv"]:
            fp = os.path.join(self.workdir, "%s.%s" %(self.name, ext))
            if os.path.isfile(fp):
                os.remove(fp)


    def loadJSON(self, filePath):
        """ Import the configurarion from a json file.

        Parameters
        ----------
        filePath : str
            Either a filename within the working directory or a full path to
            the input file.
        """

        if not os.path.isfile(filePath):
            return

        with open(filePath) as file:
            cfg = json.load(file)

        # get session configuration
        session = cfg.get('Session', None)
        if session is None or any(
                [entry not in session for entry in ["ID", "Photons"]]):
            logger.debug("Invalid configuration found for the main "
                         "settings in %s." % (filePath))
            return

        # get forward configuration
        forward = cfg.get('Forward', None)
        if forward is None or any(
                [entry not in forward for entry in ["T0", "T1", "Dt"]]):
            logger.debug("Invalid configuration found for the forward "
                         "settings in %s." % (filePath))
            return

        # get optode configuration
        optode = cfg.get('Optode', None)
        if optode is None or "source" not in optode or any(
                [entry not in optode['source']
                 for entry in ["Type", "Pos", "Dir"]]):
            logger.debug("Invalid configuration found for the optode or source "
                         "settings in %s." % (filePath))
            return

        # get domain configuration
        domain = cfg.get('Domain', None)
        if domain is None or any(
                [entry not in domain for entry in ["Dim", "Media"]]):
            logger.debug("Invalid configuration found for the domain "
                         "settings in %s." % (filePath))
            return

        # get material configuration
        media = domain.get('Media', None)
        if not isinstance(media, list) or any([
            any([sub_entry not in entry
                 for sub_entry in ["mua", "mus", "g", "n"]])
            for entry in media]):
            logger.debug("Invalid configuration found for the media "
                         "settings in %s." % (filePath))
            return

        # apply session configuration
        self.name = session.get('ID')
        self.seed = session.get('RNGSeed', -1)

        self.source['nphoton'] = session.get('Photons')

        self.boundary['missmatch'] = session.get('DoMismatch', True)
        self.boundary['specular'] = session.get('DoSpecular', True)

        self.output['normalize'] = session.get('DoNormalize', True)
        self.output['format'] = session.get('OutputFormat', "mc2")
        self.output['type'] = session.get('OutputType', "X")

        mask = session.get('SaveDataMask', "D")
        if session.get('DoPartialPath', True) and "P" not in mask:
            mask += 'P'
        if session.get('DoDCS', False) and 'M' not in mask:
            mask += 'M'
        if session.get('DoSaveExit', False) and 'X' not in mask:
            mask += 'X'
        if session.get("DoSaveExit", False) and 'V' not in mask:
            mask += 'V'
        self.output['mask'] = mask

        # apply forward configuration
        self.forward['t0'] = forward.get('T0', 0)
        self.forward['t1'] = forward.get('T1', 5e-9)
        self.forward['dt'] = forward.get('Dt', 5e-9)
        self.forward['ntime'] = int(
            (self.forward['t1'] - self.forward['t0']) // self.forward['dt']
        )

        self.boundary['n0'] = forward.get('N0', 1.)

        # apply optode configuration
        source = optode.get('Source')
        self.source['type'] = source.get('Type', "pencil")
        self.source['pos'] = source.get('Pos', [30, 30, 0])
        self.source['dir'] = source.get('Dir', [0, 0, 1])

        detector = optode.get("Detector", None)
        self.detector.clear()
        if detector is not None and isinstance(detector, list):
            for id, det in enumerate(detector):
                name = "det%d" % id
                self.detector[name] = ordered_dict({
                    'id': id,
                    'pos': det.get("Pos", [25, 30, 0]),
                    'radius': det.get("R", 1),
                })

        # apply domain configuration
        domain = cfg.get('Domain', None)

        shape = domain.get('Dim')
        # set empty domain filled with zeros
        self.domain['vol'] = np.zeros(shape, dtype='<B', order='F')

        self.domain['origin'] = cfg.get('OriginType', 0)
        self.domain['scale'] = cfg.get('LengthUnit', 1.)

        # apply material configuration
        self.material.clear()
        for i, mat in enumerate(media):
            name = "mat%d" % i
            self.material[name] = ordered_dict({
                'tag': i,
                'mua': mat['mua'],
                'mus': mat['mus'],
                'g': mat['g'],
                'n': mat['n']
            })


    def loadResults(self):
        """ Explicit load of simulation results from the working directory. """

        # retrieve results for fluence field
        filePath = os.path.join(self.workdir, self.name + ".mc2")
        if os.path.isfile(filePath):
            shape = self.domain['vol'].shape + (self.forward['ntime'], )
            logger.debug("Reading file {} with shape {}.".format(
                filePath, shape))
            store = loadmc2(filePath, shape)
            self.fluence = store.asarray()  # four-dimensional array
        else:
            logger.debug("File {} not found.".format(filePath))
            self.fluence = None


        # retrieve results for detected photons
        filePath = os.path.join(self.workdir, self.name + ".mch")
        if os.path.isfile(filePath):
            logger.debug("Reading file {}.".format(filePath))
            store = loadmch(filePath)
            self.detectedPhotons = store.asDataFrame()  # pandas dataframe
        else:
            logger.debug("File {} not found.".format(filePath))
            self.detectedPhotons = None


    def loadVolume(self, filePath, shape):
        """ Load a volume from an external file.

        Parameters
        ----------
        filePath : str
            The path to the volume input file in binary format (uint8).
        shape : tuple or list of int
            The shape of the volume array.
        """
        if not os.path.isfile(filePath):
            return

        buffer = np.fromfile(filePath, dtype='<B', count=int(np.prod(shape)))
        if len(buffer) == int(np.prod(shape)):
            logger.debug("Load volume file {} with shape {}.".format(
                filePath, shape))
            self.domain['vol'] = buffer.reshape(shape, order='F')
            self.domain['shape'] = self.domain['vol'].shape
        else:
            self.domain['vol'] = None
            self.domain['shape'] = tuple()


    def parseStats(self, message):
        """ Extract the simulation statistics from message.

        Parameters
        ----------
        message : str
            The string to be parsed for the statistical information including
            the normalization factor, number of photons, threads, the runtime,
            total energy, and absorbed energy.
        """
        field_0 = self.parseValue("normalization factor alpha", message)
        field_1 = self.parseValue("simulated", message)
        field_2 = self.parseValue("MCX simulation speed", message)
        field_3 = self.parseValue("total simulated energy", message)

        self.stat["normalizer"] = field_0
        self.stat["nphoton"] = field_1[0] if isinstance(
            field_1, list) and len(field_1) > 0 else None
        self.stat["nthread"] = field_1[2] if isinstance(
            field_1, list) and len(field_1) > 2 else None
        self.stat["runtime"] = field_2
        self.stat["energytot"] = field_3[0] if isinstance(
            field_3, list) and len(field_3) > 0 else None
        self.stat["energyabs"] = field_3[0] * field_3[1] / 100 if isinstance(
            field_3, list) and len(field_3) > 1 else None


    @staticmethod
    def parseValue(tag, string):
        """Evaluate a tagged value from a string

        Parameters
        ----------
        tag : str
            The identifier for the value.
        string : str
            The input string.

        Returns
        -------
        int, float, list
            The evaluated value after the tag. If no value was found, the
            function returns None.
        """
        match = re.search(r'%s(.*)' % tag, string)
        if not match:
            return None

        sval = match.group(1)
        regex = r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?'
        match = re.findall(regex, sval)

        n = len(match)
        if n > 1:
            return [literal_eval(s) for s in match]
        elif n == 1:
            return literal_eval(match[0])
        else:
            return None


    def run(self, thread="auto", **flags):
        """ Execute the simulation with optional flags.

        Parameters
        ----------
        thread : str or int, optional
            The number of total threads. Default is auto.
        debug : int or str, optionally
            Debug flags. If integer, must be positive; if string, must be made
            of any combinations of:

                - 1 or 'R' debug RNG
                - 2 or 'M' store photon trajectory info (saved in a .mct file)
                - 4 or 'P' print progress bar

        **flags : dict
            Additional flags with the prepended '-' or '--' being skipped.
            See mcx help for more information.
        """
        filePath = ordered_dict({
            'config': os.path.join(self.workdir, self.name + ".json"),
            # 'volume': os.path.join(self.workdir, self.name + ".mcv"),
            'fluenc': os.path.join(self.workdir, self.name + ".mc2"),
            'detect': os.path.join(self.workdir, self.name + ".mch"),
        })
        # remove any existing files
        for fp in filePath.values():
            if os.path.isfile(fp):
                os.remove(fp)

        self.dumpJSON()

        cmdItems = []  # list of executable and options to construct command

        # binary
        mcx = findMCX()
        if mcx is None:
            print("Warning: Could not find path to mcx binary.")

            if sys.platform == "win32":
                mcx = "mcx.exe"
            else:
                mcx = "mcx"
        cmdItems.append(mcx)

        # number of threads
        if not 'A' in flags.keys() and not 'autopilot' in flags.keys():
            flags['A'] = 1 if thread == "auto" else 0
        if isinstance(thread, int):
            flags['thread'] = thread

        # boundary conditions
        if not 'B' in flags.keys() and not 'bc' in flags.keys():
            flags['bc'] = self.boundary['bc']

        # add options and flags to command
        for key, val in flags.items():
            if len(key) > 1:
                cmdItems.append("--" + key)
            else:
                cmdItems.append("-" + key)
            cmdItems.append(str(val))


        # input file
        cmdItems.append("-f")
        cmdItems.append(filePath['config'])


        # run simulation
        proc = subprocess.Popen(cmdItems, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)

        # def wrap_text(msg, nchar=79):
        #     wrapped_lines = []
        #     for line in re.findall(r'.{1,%s}(?:\s+|$)' % (nchar - 4), msg):
        #         striped_line = line.strip()
        #         # (striped_line.ljust(nchar - 4)
        #         wrapped_lines.append("# %s  \n" % (striped_line))
        #     return "".join(wrapped_lines)


        if 'debug' in flags and flags['debug'] != 0:
            # print(f'{cmdItems[0]} ' + ' '.join(cmdItems[1:]))
            print(cmdItems[0] + ' '.join(cmdItems[1:]))
            print("#" * 79)
            # print("# Monte Carlo eXtreme (MCX) "
            #       "- Copyright (c) 2009-2020 Qianqian Fang")
            i = 0
            output = ""
            for line in iter(proc.stdout.readline, ""):
                output += line
                if i > 1 and i < 10 :
                    sys.stdout.write(line)
                    # sys.stdout.write("# " + str.strip(line[1:-1]))
                elif i > 14 and i !=20 and line not in ['\n', '\r\n']:
                    # sys.stdout.write(wrap_text(line))
                    sys.stdout.write(line)
                i += 1
            print("#" * 79)
            (_, errors) = proc.communicate()
        else:
            (output, errors) = proc.communicate()

        # sys.stderr.write(proc.stderr.read())
        # print(errors)

        # extract simulation statistics from output.
        self.parseStats(output)

        # retrieve results for fluence field and detected photons
        self.loadResults()


    def setBoundary(self, specular=True, missmatch=True,
                    bc="______000000", n0=1):
        """ Set the boundary conditions for the domain.

        Parameters
        ----------
        specular : bool, optional
            Enables or disables the specular reflection. By default, initial
            specular reflection is considered, thus, photon loses a small
            fraction of energy when entering the domain.
        missmatch : bool, optional
            Enable or disable reflections at the boundaries. By default, mcx
            considers refractive index mismatch at the boundaries and photons
            will be either reflected or transmitted at the boundaries based on
            the Fresnel's equation.
        bc : str, optional
            Specifies the per-face boundary conditions.  The first 6 letters
            define the boundary condition for bounding box faces at
            (xmin, ymin, zmin, xmax, ymax, zmax) axes. The 7th-12th letters can
            be either '0' to not use this face to detect photon or '1' to use
            this face for photon detection. Supported boundary conditions are:

                - '_' undefined, fallback to -b
                - 'r' like -b 1, Fresnel reflection BC
                - 'a' like -b 0, total absorption BC
                - 'm' mirror or total reflection BC
                - 'c' cyclic BC, enter from opposite face

            Default is "______000000".
        n0 : float, optional
            The background refractive index. Default is 1.
        """
        self.boundary['specular'] = specular
        self.boundary['missmatch'] = missmatch
        self.boundary['bc'] = bc
        self.boundary['n0'] = n0


    def setDetector(self, id, pos, radius, param1=None, param2=None):
        """ Set or modifies the configuration of a detector.

        Parameters
        ----------
        id : int
            Identifiyer for detector.
        pos : float
            The grid position of a detector, in grid unit.
        radius : float
            The grid position of a detector, in grid unit.
        """
        name = "det%d" % id
        if name in self.detector.keys():
            self.detector[name]['pos'] = pos
            self.detector[name]['radius'] = radius


    def setDomain(self, vol, originType=0, lengthUnit=1):
        """ Set the domain.

        Parameters
        ----------
        vol : np.ndarray of uint8
            The domain volume. A tow-dimensional or three-dimensional array.
        originType : int, optional
            Defines the coordinate origin mode of the volume. A value of 0
            assumes the lower-bottom corner of the first voxel as [1 1 1]. A
            value of 1 assumes the lower-bottom corner of the first voxel as
            [0 0 0]. Default is 0.
        lengthUnit : float, optional
            Set the edge length, in mm, of a voxel in the volume. E.g. if the
            volume used in the simulation is 0.1x0.1x0.1 mm^3, then, one should
            use a value of 0.1. Default is 1.
        """
        # mcx requires elements of volume array as little-endian and in F-order
        self.domain['vol'] = np.array(vol, dtype='<B', order='F')
        self.domain['shape'] = self.domain['vol'].shape
        self.domain['origin'] = originType
        self.domain['scale'] = lengthUnit
        self.dumpVolume()


    def setForward(self, t0, t1, dt):
        """ Set the time domain properties.

        Parameters
        ----------
        t0 : float
            The start time of the simulation, in seconds.
        t1 : float
            The end time of the simulation, in seconds.
        dt : float
            The width of each time window, in seconds.
        """
        self.forward['t0'] = t0
        self.forward['t1'] = t1
        self.forward['dt'] = dt
        self.forward['ntime'] = int((t1 - t0) // dt)


    def setMaterial(self, tag, mua, mus, g, n):
        """ Set or modifies material properties with a specific tag.

        Parameters
        ----------
        tag : int
            Identifiyer for material and associated with the the voxel value of
            the geometry domain.
        mua : float
            Absorption coefficient [1/mm].
        mus : float
            Scattering coefficient [1/mm].
        g : float
            Anisothropy of scattering.
        n : float
            refractive index.
        """
        name = "mat%d" % tag
        if name in self.material.keys():
            self.material[name]['mua'] = mua
            self.material[name]['mus'] = mus
            self.material[name]['g'] = g
            self.material[name]['n'] = n


    def setMissmatch(self, enable=1):
        """ Configures the reflections at the boundaries.

        Parameters
        ----------
        enable : bool
            Enable or disable reflections at the boundaries. By default, mcx
            considers refractive index mismatch at the boundaries and photons
            will be either reflected or transmitted at the boundaries based on
            the Fresnel's equation.
        """
        self.boundary['missmatch'] = enable


    def setOutput(self, type="X", normalize=True, mask="DSPMXVW", format="mc2"):
        """ Set the output properties of the simulation.

        Parameters
        ----------
        type : char or str, optional
            Specifies the type of data to be saved in the volumetric output.
            The supported formats include:

                - 'X' time-resolved fluence rate (1/mm^2), i.e. TPSF
                - 'F' time-resolved fluence rate integrated in each time-gate,
                - 'E' energy deposit at each voxel (normalized or unnormalized)
                - 'J' Jacobian (replay mode),
                - 'P' scattering event counts at each voxel (replay mode only)
                - 'M' partial momentum transfer

        normalize : bool
            Enables or disables to normalize the solutions. Default is True.
        mask : str, optional
            Specifies the parameters to be saved for detected photons.  The
            presence of a letter denotes that the corresponding detected photon
            data is saved, otherwise, it is not saved. The supported fields are
            listed below (the number of data columns of each field is shown
            in the parentheses):

                - 'D' detector ID
                - 'S' partial scat. even counts (#media)
                - 'P' partial path-lengths (#media)
                - 'M' momentum transfer (#media)
                - 'X' exit position (3)
                - 'V' exit direction (3)
                - 'W'  initial weight (1)

        format : str, optional
            Specifies the volumetric data output format:

                - 'mc2' MCX mc2 format (binary 32bit float) (default)
                - 'nii' Nifti format (fluence after taking log10())
                - 'jnii' JNIfTI format (http://openjdata.org)
                - 'bnii' Binary JNIfTI (http://openjdata.org)
                - 'hdr' Analyze 7.5 hdr/img format
                - 'tx3' GL texture data for rendering (GL_RGBA32F)

        """
        self.output['type'] = type
        self.output['normalize'] = normalize
        self.output['mask'] = mask.upper()
        self.output['format'] = format


    def setSeed(self, seed):
        """ Set the seed of the CPU random number generator.

        Parameters
        ----------
        seed : int
            The seed of the CPU random number generator (RNG). A value of -1
            let MCX to automatically seed the CPU-RNG using system clock. A
            value n > 0 sets the CPU-RNG's seed to n.
        """
        self.seed = seed


    def setSource(self, nphoton, pos, dir):
        """ Set the photon source.

        Parameters
        ----------
        nphoton : int
            The total number of photons to be simulated.
        pos : list, ndarray
            The grid position of a source, can be non-integers, in grid unit.
        dir : list, ndarray
            The unitary directional vector of the photon at launch.
        """
        self.source['nphoton'] = nphoton
        self.source['pos'] = pos
        self.source['dir'] = dir


    def setSourceType(self, type, param1=None, param2=None, pattern=None):
        """ Set the photon source.

        Parameters
        ----------
        type : str
            The Source type. Must be one of the following: pencil, isotropic,
            cone, gaussian, planar, pattern, fourier, arcsine, disk.
        param1 : list, ndarray
            Source parameters, 4 floating-point numbers.
        param2 : list, ndarray
            Additional source parameters, 4 floating-point numbers.
        """
        self.source['type'] = type
        self.source['param1'] = param1
        self.source['param2'] = param2
        self.source['pattern'] = pattern


    def setSpecular(self, enable=1):
        """ Configures the specular reflection at the initial entry of
        the photons to the domain (entry from a 0-voxel to a non-zero voxel).

        Parameters
        ----------
        enable : bool
            Enables or disables the specular reflection. By default, initial
            specular reflection is considered, thus, photon loses a small
            fraction of energy, but enter the domain.
        """
        self.boundary['specular'] = enable


