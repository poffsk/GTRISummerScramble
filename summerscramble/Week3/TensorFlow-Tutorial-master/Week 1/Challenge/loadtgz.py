#! /usr/bin/env python3.6

import json
import tarfile

import click
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from scipy.sparse import csc_matrix


def load_json_from_tarball(datafile):
    with tarfile.open(datafile, 'r') as f:
        mbrs = f.getmembers()
        # print('Members are {} and {}'.format(mbrs[0],mbrs[1]))
        # assert_that(len(mbrs)).is_equal_to(1)
        if (len(mbrs) == 1):
            return json.load(f.extractfile(mbrs[0].name))
        else:
            return json.load(f.extractfile(mbrs[1].name))




def sparse_matrix_from_dict(d):
    """Return a scipy sparse matrix from ACIS data dict

    With sparse matrices, you usually have to choose whether it will
    be fast to access the data by column or row.  Here we use the
    csc_matrix to facilitate iterating over the columns.
    """

    return csc_matrix(
        (d['data'], (d['row_index'], d['column_index'])),
        shape=tuple(d['shape']))


def show_label_counts(d):
    unique, counts = np.unique(d['labels'], return_counts=True)
    click.echo('counts for each label: {}'.format(
        list(zip(unique, counts))))


def find_data_range(d):
    datamin = min(d['data'])
    datamax = max(d['data'])
    click.echo('Range of data: min is {}, max is {}'.format(datamin, datamax))
    return datamin, datamax


#@click.command()
#@click.argument(
  #  'data', type=click.Path(exists=True)
#)
def main(data):
    d = load_json_from_tarball(data)
    click.echo('keys in data dictionary: {}'.format(d.keys()))

    show_label_counts(d)

    m = sparse_matrix_from_dict(d)

    click.echo(
        'shape:{} number_non_zero:{} proportion_non_zero:{}'.format(
            m.shape, m.nnz, m.nnz / np.prod(m.shape)))

    col_nz_counts = np.array(
        [m[:, j].count_nonzero() for j in range(m.shape[1])])

    nz_val_counts = np.bincount(col_nz_counts)
    assert np.sum(col_nz_counts == 1) == nz_val_counts[1]
    click.echo(
        'for each number of non-zero values per column '
        'show the number of columns with that number; e.g., '
        'first value is the number of columns with no '
        'non-zero values at all, second value printed '
        'is the number of columns with one non-zero value, etc.'
        '\n{}'.format(
            nz_val_counts[:19]))

    click.echo(
        'number of columns with fewer than 100 non-zero values'
        ': {}'.format(
            np.sum(col_nz_counts < 100)))
    click.echo(
        'number of columns with at least 100 non-zero values'
        ': {}'.format(
            np.sum(col_nz_counts >= 100)))

    # set up counts for histogram, but use logs to see better,
    # and add a smidge so that we can see zeroes off to the left
    x = np.log10(col_nz_counts + 0.0000000001)


    #trying to figure stuff out
    #plot(go.Figure(
       # data=[go.Histogram(x=x, nbinsx=50)],
        #layout=go.Layout(title='log10 nonzero counts for columns')))

    
    print(d['data'][0:5])

if __name__ == '__main__':
    boogie = 'training_data.tgz'
    #boogie2 = 'C:\\Users\\camil\\Documents\\GTRI Internship\\training_data.tgz'
    main(boogie)



