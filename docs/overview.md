This page provides a general overview of the purpose and functionality of TAAT.

### Introduction

TAAT was conceived as a software toolkit aimed at researchers working with digitised tape archives, with the purpose of assisting such researchers - be they musicologists, historians, librarians, archivists, or composers - to easily and rapidly sort, connect, and categorise large collections of audio material from tape sources.

The toolkit helps to determine:

*   where relationships exist between segments of audio files;
*   what the differences are between two related audio recordings.

In particular, TAAT was designed to address the shortcomings of existing tools, which fail to deal effectively with issues the process of tape archive digitisation invariably introduces, such as pitch fluctuations, tone saturation and frequency filtering.

### How it works

The TAAT analysis engine works by first creating a timbral profile of its input audio files. Each file is processed in chunks, the size of which can be determined by the use. The chunks are then compared using cross similarity matrices and [recurrence quantification analysis](https://en.wikipedia.org/wiki/Recurrence_quantification_analysis), in order to establish a similarity score. Sequences of chunk scores represent similar timbral profiles that might inform a consistent musical progression. Results can then be output to the terminal, plotted as charts, or saved to a `JSON` file which can be interpreted further by other applications or programs.
