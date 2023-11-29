import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib as plt
import math as m

x = np.arange(0, 9, 0.3)
print(x)
y = m.sin(x)*m.sqrt(x)

plt.plot(x, y, 'r--')
"""

axes

Add an Axes to the current figure and make it the current Axes.

cla

Clear the current axes.

clf

Clear the current figure.

close

Close a figure window.

delaxes

Remove an Axes (defaulting to the current axes) from its figure.

fignum_exists

Return whether the figure with the given id exists.

figure

Create a new figure, or activate an existing figure.

gca

Get the current Axes.

gcf

Get the current figure.

get_figlabels

Return a list of existing figure labels.

get_fignums

Return a list of existing figure numbers.

sca

Set the current Axes to ax and the current Figure to the parent of ax.

subplot

Add an Axes to the current figure or retrieve an existing Axes.

subplot2grid

Create a subplot at a specific location inside a regular grid.

subplot_mosaic

Build a layout of Axes based on ASCII art or nested lists.

subplots

Create a figure and a set of subplots.

twinx

Make and return a second axes that shares the x-axis.

twiny

Make and return a second axes that shares the y-axis.

Adding data to the plot
Basic
plot

Plot y versus x as lines and/or markers.

errorbar

Plot y versus x as lines and/or markers with attached errorbars.

scatter

A scatter plot of y vs.

plot_date

[Discouraged] Plot coercing the axis to treat floats as dates.

step

Make a step plot.

loglog

Make a plot with log scaling on both the x- and y-axis.

semilogx

Make a plot with log scaling on the x-axis.

semilogy

Make a plot with log scaling on the y-axis.

fill_between

Fill the area between two horizontal curves.

fill_betweenx

Fill the area between two vertical curves.

bar

Make a bar plot.

barh

Make a horizontal bar plot.

bar_label

Label a bar plot.

stem

Create a stem plot.

eventplot

Plot identical parallel lines at the given positions.

pie

Plot a pie chart.

stackplot

Draw a stacked area plot.

broken_barh

Plot a horizontal sequence of rectangles.

vlines

Plot vertical lines at each x from ymin to ymax.

hlines

Plot horizontal lines at each y from xmin to xmax.

fill

Plot filled polygons.

polar

Make a polar plot.

Spans
axhline

Add a horizontal line across the Axes.

axhspan

Add a horizontal span (rectangle) across the Axes.

axvline

Add a vertical line across the Axes.

axvspan

Add a vertical span (rectangle) across the Axes.

axline

Add an infinitely long straight line.

Spectral
acorr

Plot the autocorrelation of x.

angle_spectrum

Plot the angle spectrum.

cohere

Plot the coherence between x and y.

csd

Plot the cross-spectral density.

magnitude_spectrum

Plot the magnitude spectrum.

phase_spectrum

Plot the phase spectrum.

psd

Plot the power spectral density.

specgram

Plot a spectrogram.

xcorr

Plot the cross correlation between x and y.

Statistics
ecdf

Compute and plot the empirical cumulative distribution function of x.

boxplot

Draw a box and whisker plot.

violinplot

Make a violin plot.

Binned
hexbin

Make a 2D hexagonal binning plot of points x, y.

hist

Compute and plot a histogram.

hist2d

Make a 2D histogram plot.

stairs

A stepwise constant function as a line with bounding edges or a filled plot.

Contours
clabel

Label a contour plot.

contour

Plot contour lines.

contourf

Plot filled contours.

2D arrays
imshow

Display data as an image, i.e., on a 2D regular raster.

matshow

Display an array as a matrix in a new figure window.

pcolor

Create a pseudocolor plot with a non-regular rectangular grid.

pcolormesh

Create a pseudocolor plot with a non-regular rectangular grid.

spy

Plot the sparsity pattern of a 2D array.

figimage

Add a non-resampled image to the figure.

Unstructured triangles
triplot

Draw an unstructured triangular grid as lines and/or markers.

tripcolor

Create a pseudocolor plot of an unstructured triangular grid.

tricontour

Draw contour lines on an unstructured triangular grid.

tricontourf

Draw contour regions on an unstructured triangular grid.

Text and annotations
annotate

Annotate the point xy with text text.

text

Add text to the Axes.

figtext

Add text to figure.

table

Add a table to an Axes.

arrow

Add an arrow to the Axes.

figlegend

Place a legend on the figure.

legend

Place a legend on the Axes.

Vector fields
barbs

Plot a 2D field of barbs.

quiver

Plot a 2D field of arrows.

quiverkey

Add a key to a quiver plot.

streamplot

Draw streamlines of a vector flow.

Axis configuration
autoscale

Autoscale the axis view to the data (toggle).

axis

Convenience method to get or set some axis properties.

box

Turn the axes box on or off on the current axes.

grid

Configure the grid lines.

locator_params

Control behavior of major tick locators.

minorticks_off

Remove minor ticks from the Axes.

minorticks_on

Display minor ticks on the Axes.

rgrids

Get or set the radial gridlines on the current polar plot.

thetagrids

Get or set the theta gridlines on the current polar plot.

tick_params

Change the appearance of ticks, tick labels, and gridlines.

ticklabel_format

Configure the ScalarFormatter used by default for linear Axes.

xlabel

Set the label for the x-axis.

xlim

Get or set the x limits of the current axes.

xscale

Set the xaxis' scale.

xticks

Get or set the current tick locations and labels of the x-axis.

ylabel

Set the label for the y-axis.

ylim

Get or set the y-limits of the current axes.

yscale

Set the yaxis' scale.

yticks

Get or set the current tick locations and labels of the y-axis.

suptitle

Add a centered suptitle to the figure.

title

Set a title for the Axes.

Layout
margins

Set or retrieve autoscaling margins.

subplots_adjust

Adjust the subplot layout parameters.

subplot_tool

Launch a subplot tool window for a figure.

tight_layout

Adjust the padding between and around subplots.

Colormapping
clim

Set the color limits of the current image.

colorbar

Add a colorbar to a plot.

gci

Get the current colorable artist.

sci

Set the current image.

get_cmap

Get a colormap instance, defaulting to rc values if name is None.

set_cmap

Set the default colormap, and applies it to the current image if any.

imread

Read an image from a file into an array.

imsave

Colormap and save an array as an image file.
"""