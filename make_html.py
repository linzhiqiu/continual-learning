import dominate
from dominate.tags import *
import os

def make_index_html(links, descriptions, href='temp.html'):
    doc = dominate.document(title='Index HTML')
    
    with doc.head:
        link(rel='stylesheet', href='html/style.css')
    
    with doc:
        h1('Index Page')
        with div(id='header').add(ol()):
            for i, d in zip(links, descriptions):
                li(a(d.title(), href=i))
    # print(doc)
    with open(href, "w+") as f:
        f.write(doc.render())

from html4vision import Col, imagetable

def make_table_html(headers,
                    cols,
                    summary_row=None,
                    sortcol=None,
                    href="temp.html",
                    image_col_name=None,
                    html_title='Sorting Example'):
    all_cols = [
        Col('id0', 'ID'), # 0-based indexing
    ]
    for h, col in zip(headers, cols):
        type_str = 'img' if h == image_col_name else 'text'
        all_cols.append(Col(type_str, h, col))

    header_sort_index = headers.index(sortcol) + 1
    imagetable(all_cols, href, html_title,
        summary_row=[""]+summary_row,    # add a summary row showing overall statistics of the dataset
        summary_color='#fff9b7',    # highlight the summary row
        imscale=0.4,                # scale images to 0.4 of the original size
        sortcol=header_sort_index,
        sortable=True,              # enable interactive sorting
        sticky_header=True,         # keep the header on the top
        sort_style='materialize',   # use the theme "materialize" from jquery.tablesorter
        zebra=True,                 # use zebra-striped table
        pathrep=None,
        # precompute_thumbs=True,
        # thumb_quality=95,
        # style
        # imsize=1,
        # preserve_aspect=True
    )

# make_index_html(['temp.html'], ['Index page'])