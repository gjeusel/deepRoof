import csv
import os
from os.path import isfile
from argparse import ArgumentParser
import re

from osgeo import gdal, osr, ogr, gdalconst
import numpy as np
from PIL import Image


DEFAULT_URL = 'https://api.mapbox.com/v4/mapbox.satellite/${z}/${x}/${y}.png?access_token=<TOKEN>'

WMS_TEMPLATE = \
'''<GDAL_WMS>
    <Service name="TMS">
        <ServerUrl>{server_url}</ServerUrl>
    </Service>
    <Referer>referer</Referer>
    <DataWindow>
        <UpperLeftX>-20037508.34</UpperLeftX>
        <UpperLeftY>20037508.34</UpperLeftY>
        <LowerRightX>20037508.34</LowerRightX>
        <LowerRightY>-20037508.34</LowerRightY>
        <TileLevel>19</TileLevel>
        <TileCountX>1</TileCountX>
        <TileCountY>1</TileCountY>
        <YOrigin>top</YOrigin>
    </DataWindow>
    <Projection>EPSG:3857</Projection>
    <BlockSizeX>256</BlockSizeX>
    <BlockSizeY>256</BlockSizeY>
    <BandsCount>3</BandsCount>
</GDAL_WMS>
'''


def convert(a_source, b_source, transform):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(a_source, b_source)

    point.Transform(transform)

    point_wkt = point.ExportToWkt()
    a_target, b_target = re.match(r'POINT \((-?[0-9]*\.?[0-9]*) (-?[0-9]*\.?[0-9]*) 0\)', point_wkt).groups()
    a_target = float(a_target)
    b_target = float(b_target)

    return a_target, b_target


def WGS84toWebMercator(lon, lat):
    """Convert from WGS84 to WebMercator."""
    return convert(lon, lat, transform_WGS84_to_WebMercator)


def coord2pix(x, y):
    """Convert from WebMercator to pixel index."""
    pix_x = int(round((x - originX) / pixelSizeX))
    pix_y = int(round((y - originY) / pixelSizeY))
    return pix_x, pix_y


def fetch_box(west, east, north, south, border):
    """Fetch a box (rectangle) from WGS84 coordinates."""
    assert (west < east)
    assert (north > south)

    x_webmercator_min, y_webmercator_max = WGS84toWebMercator(west, north)
    x_webmercator_max, y_webmercator_min = WGS84toWebMercator(east, south)

    assert (x_webmercator_min < x_webmercator_max)
    assert (y_webmercator_min < y_webmercator_max)

    x_pix_min, y_pix_min = coord2pix(x_webmercator_min, y_webmercator_max)
    x_pix_max, y_pix_max = coord2pix(x_webmercator_max, y_webmercator_min)

    assert (x_pix_min < x_pix_max)
    assert (y_pix_min < y_pix_max)

    # add a border
    x_pix_min -= border
    y_pix_min -= border
    x_pix_max += border
    y_pix_max += border

    size_x = x_pix_max - x_pix_min + 1
    size_y = y_pix_max - y_pix_min + 1

    image = dataset.ReadAsArray(x_pix_min, y_pix_min, size_x, size_y)
    image = np.rollaxis(image, 0, 3)

    return image



def download(image_id, image_file_name, min_lon, max_lon, max_lat, min_lat, border=10):
    np_image = fetch_box(min_lon, max_lon, max_lat, min_lat, border)
    pillow_image = Image.fromarray(np_image)
    with open(image_file_name, 'wb') as out_file:
    	pillow_image.save(out_file, 'JPEG')



if __name__ == '__main__':
	class Arguments:
		pass

	parser = ArgumentParser(description='Download images for the SolarMap challenge')
	parser.add_argument('-b', '--bounding_boxes', action='store', required=True)
	parser.add_argument('-o', '--output-dir', action='store', required=True)
	parser.add_argument('-w', '--overwrite', action='store_true')
	parser.add_argument('-u', '--url-template', action='store', default=DEFAULT_URL)
	parser.add_argument('-a', '--access-token', action='store', required=True)
	parser.add_argument('-t', '--tmp-dir', action='store', default='/tmp')

	args = Arguments()
	parser.parse_args(namespace=args)

	print('Reading bounding boxes from {input}, storing into {output}'.format(input=args.bounding_boxes, output=args.output_dir))
	if args.overwrite:
		print('Will overwrite existing files')

	print('Server URL template is {url}'.format(url=args.url_template))
	server_url = args.url_template.replace('<TOKEN>', args.access_token)
	wms = WMS_TEMPLATE.format(server_url=server_url)
	vrt_file_name = os.path.join(args.tmp_dir, 'download.vrt')
	with open(vrt_file_name, 'w') as vrt_file:
		vrt_file.write(wms)

	gdal.UseExceptions()

	# Open virtual dataset
	dataset = gdal.Open(vrt_file_name, gdalconst.GA_ReadOnly)
	originX, pixelSizeX, _, originY, _, pixelSizeY = dataset.GetGeoTransform()

	# Create coordinates transformation from WGS84 to WebMercator
	source = osr.SpatialReference()
	source.ImportFromEPSG(4326)     # WGS84
	target = osr.SpatialReference()
	target.ImportFromEPSG(3857)     # WebMercator
	transform_WGS84_to_WebMercator = osr.CoordinateTransformation(source, target)

	# Now read file and download images
	with open(args.bounding_boxes, 'r') as in_file:
		csvreader = csv.reader(in_file)
		
		# Skip header
		next(csvreader)

		# Read remainder of file
		for row in csvreader:
			image_id, min_lon, max_lon, max_lat, min_lat = [row[0]] + list(map(float, row[1:]))
			image_file_name = os.path.join(args.output_dir, image_id + '.jpg')

			if not isfile(image_file_name) or args.overwrite:
				print('[DOWNLOAD] {file}'.format(file=image_file_name))
				download(image_id, image_file_name, min_lon, max_lon, max_lat, min_lat)
			else:
				print('  [IGNORE] {file}'.format(file=image_file_name))

	print('Done!')

