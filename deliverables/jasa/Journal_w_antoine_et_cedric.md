# JOURNAL with Antoine and Cedric [ 27 | 01 | 2021 ]

## Guidelines

- [ ] Something straight to the point, and then I could include it and work on it depending on what we did.
- [ ] a word/ref to the sine sweep technique used

## Actual text

To evaluate the generalizability of the proposed approach to real measured RIR, we used the dECHORATE dataset [1,currently under review].
This dataset consists of 1320 RIR measurements made in a 6 m $\times$
6 m $\times$ 2.4 m acoustic room in the Acoustic Lab of the Bar-Ilan University.
The walls and ceilings' absorption properties can be changed by flipping double-sided panels with one reflective and one absorbing face. This allows obtaining different room configuration, each characterized by different early reflections prominency and reverberation levels.

The measurements correspond to 30 receivers, 4 sources and 11 room configurations.
The receivers are organized in 6 linear arrays, each of 5 omnidirectional microphones AKG CK32 and 4 loudspeaker Avantone Mixcube are used as sources. Both arrays and sources are distributed randomly in the room.
The room configurations are created by flipping all the panels on one or more surfaces.
Among the 11 configurations provided in the datasets, the following ones are considered because ... ...

Finally, the RIRs have been estimated using the Exponential Sine Sweep technique described in~\cite{farina2007}\footnote{code and more details are available at the \url{https://github.com/Chutlhu/DechorateDB}{dataset website}}.
The ground-truth values of the absorption coefficients were computed by knowing the...

## References

1. paper under review (shall we create an arXiv or Hal entry?)
2. @inproceedings{farina2007advancements,
  title={Advancements in impulse response measurements by sine sweeps},
  author={Farina, Angelo},
  booktitle={Audio Engineering Society Convention 122},
  year={2007},
  organization={Audio Engineering Society}
}
