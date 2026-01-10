'''
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021).
Cellpose: a generalist algorithm for cellular segmentation. Nature methods,
18(1), 100-106.

S. Di Cataldo, E. Ficarra, A. Acquaviva, E. Macii (2010).
Automated segmentation of tissue images for computerized IHC analysis,
https://doi.org/10.1016/j.cmpb.2010.02.002
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2

from skimage import data, io, img_as_ubyte
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops_table

from cellpose import models
from cellpose import utils

import pandas as pd
import os
import argparse
import json
import sqlite3
import urllib.request

from scipy.spatial.distance import cdist
from scipy import ndimage
from scipy.ndimage import sobel

class cellpose():
	def __init__(self, args):

		'''CARE: download_H_img_EQ is a workaround, be aware of current status'''
		super().__init__()

		self.database_path = args.database_path
		self.images_ihc_path = args.images_ihc_path
		self.masks_ihc_path = args.masks_ihc_path
		self.masks_ihc_cells_path = args.masks_ihc_cells_path

		self.model_path = args.model_path

		self.gpu_usage = True
		self.main_model = 'cellpose'	# 'cellpose', 'stardist', 'staining' ('staining' is not deepL model)
		self.model_type = 'cyto2'  # 'cyto' for cytoplasmic masks, 'nuclei' for nuclear masks; 'cyto2'

		self.model_channels = [0, 0]	# [0,0] grayscale input [cyco, nuclei]
										# 0=grayscale, 1=red, 2=green, 3=blue

	def save_last_processed_id(self, last_processed_id, filename):
		with open(filename, 'w') as file:
			file.write(str(last_processed_id))

	def load_last_processed_id(self, filename):
		try:
			with open(filename, 'r') as file:
				last_processed_id = int(file.read())
				return last_processed_id
		except FileNotFoundError:
			return 0
	# Separate the individual stains from the IHC image

	def color_separate(self, ihc_rgb):

		# Convert the RGB image to HED using the prebuilt skimage method
		ihc_hed = rgb2hed(ihc_rgb)

		# Create an RGB image for each of the separated stains
		# Convert them to ubyte for easy saving to drive as an image
		null = np.zeros_like(ihc_hed[:, :, 0])
		ihc_h = img_as_ubyte(hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
		ihc_d = img_as_ubyte(hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))

		return (ihc_h, ihc_d)


	def create_stain_mask(self, H_img, D_img, mask_cropped):
		print(mask_cropped.shape, 'mask')
		print(H_img.shape, 'H_img')

		img = H_img
		img2 = D_img
		stain_mask_list = []
		# for i in range(2):
		# 	if i == 0:
		# 		H_cells = np.copy(img)
		# 		D_cells = np.copy(img2)
		#
		# 		H_cells[mask_cropped == 0] = 0  # convert every pixel to black which is 0
		# 		D_cells[mask_cropped == 0] = 0
		#
		# 		H_img = H_cells
		# 		D_img = D_cells
		#
		# 	elif i == 1:
		# 		H_not_cells = np.copy(img)
		# 		D_not_cells = np.copy(img2)
		#
		# 		# Extraction of cells/cyto
		# 		H_not_cells[mask_cropped > 0] = 0  # convert every pixel to black which is greater 0
		# 		D_not_cells[mask_cropped > 0] = 0
		#
		# 		H_img = H_not_cells
		# 		D_img = D_not_cells

		def calculate_histogram(image):
			hist = cv2.calcHist([image], [0], None, [256], [0, 255])
			return hist

		def normalize_histogram(hist):
			return hist / np.sum(hist)
			#return hist / hist.sum()

		def determine_threshold(hist_h, hist_d):
			# Find the point of maximum difference between the histograms
			diff = np.abs(hist_h - hist_d)
			threshold = np.argmax(diff)
			return threshold

		def classify_pixels(image_h, image_d, threshold):
			rows, cols = image_h.shape[:2]
			result_image = np.zeros((rows, cols, 3), dtype=np.uint8)
			print(threshold)

			for i in range(rows):
				for j in range(cols):
					intensity_h = image_h[i, j]
					intensity_d = image_d[i, j]
					# Create blue mask for H > D
					if intensity_h > intensity_d + threshold:
						result_image[i, j] = [intensity_h, 0, 0]
					# Create brown mask for D > H
					elif intensity_d > intensity_h + threshold:
						result_image[i, j] = [0, 0, intensity_d]
					else:
						# Assign the pixel to the image with higher intensity
						if intensity_h < intensity_d:
							result_image[i, j] = [intensity_h, 0, 0]  # Blue mask
						else:
							result_image[i, j] = [0, 0, intensity_d]  # Brown mask

			return result_image
		'''
		https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/
		OpenCV Histogram ...
		'''

		# Convert to grayscale
		image_h_gray = cv2.cvtColor(H_img, cv2.COLOR_BGR2GRAY)
		image_d_gray = cv2.cvtColor(D_img, cv2.COLOR_BGR2GRAY)

		# Compute histograms
		hist_h = calculate_histogram(image_h_gray)
		hist_d = calculate_histogram(image_d_gray)

		# Normalize histograms
		norm_hist_h = normalize_histogram(hist_h)
		norm_hist_d = normalize_histogram(hist_d)

		# plot the histogram
		plt.figure()
		plt.title("Grayscale Histogram H")
		plt.xlabel("Bins")
		plt.ylabel("Number of Pixels")
		plt.plot(hist_h)
		plt.xlim([0, 256])

		# plot the histogram
		plt.figure()
		plt.title("Grayscale Histogram D")
		plt.xlabel("Bins")
		plt.ylabel("Number of Pixels")
		plt.plot(hist_d)
		plt.xlim([0, 256])
		#plt.show()

		# Determine threshold
		threshold = determine_threshold(norm_hist_h, norm_hist_d)

		image_h_gray = np.invert(image_h_gray)
		image_d_gray = np.invert(image_d_gray)
		# image_h_gray = cv2.equalizeHist(image_h_gray)
		# image_d_gray = cv2.equalizeHist(image_d_gray)

		plt.figure()
		plt.axis("off")
		plt.title("H")
		plt.imshow(cv2.cvtColor(image_h_gray, cv2.COLOR_GRAY2RGB))

		plt.figure()
		plt.axis("off")
		plt.title("d")
		plt.imshow(cv2.cvtColor(image_d_gray, cv2.COLOR_GRAY2RGB))


		# Classify pixels
		stain_mask = classify_pixels(image_h_gray, image_d_gray, threshold)

		#
		# # Create mask for blue pixels where H_img is brighter
		# blue_mask = H_img > D_img
		# blue_color = np.stack([np.zeros_like(H_img), np.zeros_like(H_img), H_img * blue_mask], axis=-1)
		#
		# # Create mask for brown pixels where D_img is brighter
		# brown_mask = ~blue_mask
		# brown_color = np.stack([D_img * brown_mask, np.zeros_like(H_img), np.zeros_like(H_img)], axis=-1)
		#
		# # Create mask for pixels with brightness 0,0,0
		# black_mask = (H_img == 0) & (D_img == 0)
		# class_0_color = np.zeros_like(H_img)
		#
		# # Initialize stain_mask with a transparent background
		# stain_mask = np.zeros_like(blue_color, dtype=np.uint8)
		#
		# # Fill the stain_mask with colors based on blue, brown, and class 0 masks
		# stain_mask[blue_mask] = blue_color[blue_mask]
		# stain_mask[brown_mask] = brown_color[brown_mask]
		#
		# # Assign class 0 color only to pixels where the black_mask is True
		# stain_mask[..., 0][black_mask] = class_0_color[black_mask]
		# stain_mask[..., 1][black_mask] = class_0_color[black_mask]
		# stain_mask[..., 2][black_mask] = class_0_color[black_mask]



		combined_stain_mask = stain_mask
		# stain_mask_list.append(stain_mask)
		#
		# # Combine both masks
		# combined_stain_mask = stain_mask_list[0] | stain_mask_list[1]



		# Create Contours for each unique component of cell mask
		mask_cropped = mask_cropped.astype(np.uint8)

		# Create an empty image to draw contours
		contour_image = np.zeros_like(mask_cropped)

		# Get unique values in the mask
		unique_values = np.unique(mask_cropped)

		# Iterate through unique values
		for value in unique_values:
			# Create a binary mask for the current value
			component_mask = np.uint8(mask_cropped == value)

			# Find contours for the current component
			contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			# Draw a filled contour for each pixel
			for contour in contours:
				# Generate a random color for each contour
				color = tuple(map(int, np.random.randint(0, 255, size=(3,))))
				cv2.drawContours(combined_stain_mask, [contour], -1, color, thickness=2)

		return combined_stain_mask

	def download_H_img_EQ(self, img, mask, mask_filename):
		# Extraction of Epidermis
		extracted_area = np.copy(img)
		extracted_area[mask != 2] = 255  # convert every pixel to white which isn`t 2 (Epidermis)


		H, D = self.color_separate(extracted_area)

		H_img = np.invert(H[:, :, 2])
		#D_img = np.invert(D[:, :, 2])

		# H_img = cv2.cvtColor(H[:, :, :3], cv2.COLOR_BGR2GRAY)
		# D_img = cv2.cvtColor(D[:, :, :3], cv2.COLOR_BGR2GRAY)

		# Example of histogram equalization using OpenCV
		# https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
		H_img_EQ = cv2.equalizeHist(H_img)
		# D_img_EQ = cv2.equalizeHist(D_img)

		'''Save H_img to use the images for cellpose algorithm on GPU Cluster HPC'''
		save_dir = 'data/images_H_img_EQ/'

		valid_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
		if not any(mask_filename.endswith(ext) for ext in valid_extensions):
			mask_filename += '.png'  # Standard auf .png setzen

		file_path = os.path.join(save_dir, mask_filename)

		success = cv2.imwrite(file_path, H_img_EQ)

		if success:
			print(f"H_img saved {file_path}.")
		else:
			print(f"Error saving {file_path}.")

		return
		''''''

	def crop_image(self, H_img_EQ, mask, model, mask_filename):

		# Extraction of Epidermis
		extracted_area = np.copy(H_img_EQ)
		#extracted_area[mask != 2] = 255  # convert every pixel to white which isn`t 2 (Epidermis)

		# Convert the predicted_mask to uint8 type for findContours
		mask_uint8 = (mask == 2).astype(np.uint8)

		# Find contours in the mask
		contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		H = None
		D = None
		H_img = None
		D_img = None
		H_combined_mask = None
		D_combined_mask = None
		cropped_image = None
		stain_mask = None

		if contours:
			# Find the bounding box around the non-background region in the predicted mask
			x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

			# Crop the original image based on the scaled bounding box
			#cropped_image = extracted_area[y_original:y_original + h_original, x_original:x_original + w_original]
			#cropped_image = img[y_original:y_original + h_original, x_original:x_original + w_original]
			cropped_image_H = extracted_area[y:y + h, x:x + w]



			# cv2.imshow('Loaded Image', cropped_image_H)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			if self.main_model == 'cellpose':
				#H_masks, H_flows, H_styles, H_diams = model.eval(H_img, diameter=None, channels=self.model_channels)
				H_masks, _, _, _ = model.eval(cropped_image_H, diameter=None, channels=self.model_channels)
				#D_masks, D_flows, D_styles, D_diams = model.eval(D_img, diameter=None, channels=self.model_channels)
				D_masks = H_masks

			elif self.main_model == 'stardist':
				H_masks, H_details = model.predict_instances(normalize(H_img_EQ))
				D_masks, D_details = model.predict_instances(normalize(D_img_EQ))

			elif self.main_model == 'staining':
				# Create Stain Mask
				# Load numpy cell seg mask to crop it and create seperate staining
				mask_cell_path = os.path.join(self.masks_ihc_cells_path, mask_filename)
				loaded_array = np.load(mask_cell_path)
				# Access the array using the key specified during saving (in this case, 'mask')
				mask = loaded_array['mask']

				mask_cropped = mask[y:y + h, x:x + w]
				stain_mask = self.create_stain_mask(H, D, mask_cropped)

				H_img = H_img_EQ
				D_img = D_img_EQ

				H_masks = H_img
				D_masks = D_img

			# Create an empty mask and place the resized predicted mask at its original position
			H_combined_mask = np.zeros_like(H_img_EQ[:, :, 0], dtype=np.uint8)
			#D_combined_mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)

			# Ensure the dimensions match before assignment
			h_resized, w_resized = H_masks.shape
			H_combined_mask[y:y + h_resized, x:x + w_resized] = H_masks

			# # Ensure the dimensions match before assignment
			# h_resized, w_resized = D_masks.shape
			# D_combined_mask[y:y + h_resized, x:x + w_resized] = D_masks

		return H, D, H_img, D_img, H_combined_mask, D_combined_mask, cropped_image, stain_mask

	def load_last_processed_id(self, filename):
		try:
			with open(filename, 'r') as file:
				return int(file.read().strip())
		except (FileNotFoundError, ValueError):
			return 0

	def save_last_processed_id(self, last_processed_id, filename):
		with open(filename, 'w') as file:
			file.write(str(last_processed_id))

	def main(self, id_value):
		if self.main_model == 'cellpose':
			# Initialize cellpose model
			model = models.Cellpose(gpu=self.gpu_usage, model_type=self.model_type)
		elif self.main_model == 'stardist':
			from stardist.models import StarDist2D
			from stardist.data import test_image_nuclei_2d
			from stardist.plot import render_label
			from csbdeep.utils import normalize

			# prints a list of available models
			StarDist2D.from_pretrained()

			# Define a pretrained model to segment nuclei in fluorescence images (download from pretrained)
			model = StarDist2D.from_pretrained('2D_versatile_fluo')
		elif self.main_model == 'staining':
			model = None

		# Function to convert the file name to the required URL format
		def convert_to_url(file_name):
			first_part, rest = file_name.split("-", 1)  # Split on the first dash
			url = f"http://images.proteinatlas.org/{first_part}/{rest}.jpg"
			return url

		# Function to read missing file names from the text file
		def load_missing_files(file_path):
			with open(file_path, 'r') as file:
				missing_files = [line.strip() for line in file.readlines()]
			return missing_files

		missing_files = load_missing_files(missing_files_path)

		# Convert missing file names to URLs
		missing_urls = {convert_to_url(file) for file in missing_files}

		#filename = f'last_processed_id_{id_value}.txt' if id_value else 'last_processed_id.txt'

		conn = sqlite3.connect(self.database_path)
		cur = conn.cursor()

		#last_processed_id = self.load_last_processed_id(filename)

		cur.execute("SELECT mainID, name, url FROM elGenes WHERE url IN ({})".format(
		','.join('?' for _ in missing_urls)))
		# Execute the query with the missing URLs
		cur.execute(query, list(missing_urls))
		rows = cur.fetchall()
		conn.close()

		all_image_url = []
		for row in rows:
			mainID, name, url = row
			print(name)
			all_image_url.append((mainID, name, url))
		# Processing images
		skipped_images = []
		renamed_files = []
		for mainID, name, url in all_image_url:
			try:
				print(mainID,'mainID')
				# specific image
				# if name != 'TP63':
				# 	continue


				# mask
				mask_filename = url.split('/')[-2] + '-' + url.split('/')[-1]
				mask_filename = mask_filename.split('.')[0]
				mask_path = f'data/masks_IHC/{mask_filename}.npz'
				print(mask_path)
				loaded_array = np.load(mask_path)
				# Access the array using the key specified during saving (in this case, 'mask')
				mask = loaded_array['mask']

				'''CARE: download_H_img_EQ is a workaround, be aware of current status'''
				download_H_img_EQ = False

				if download_H_img_EQ:	# if Internet Connection is possible (outside of Cluster)
					req = urllib.request.urlopen(url)
					arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
					ihc_img = cv2.imdecode(arr, -1)
					input_image = cv2.cvtColor(ihc_img, cv2.COLOR_BGR2RGB)

					self.download_H_img_EQ(input_image, mask, mask_filename)

				else:
					H_img_EQ_path = f'data/images_H_img_EQ/{mask_filename}.png'
					#print(H_img_path, 'H_img_Path')
					input_image = cv2.imread(H_img_EQ_path)


					H, D, H_img, D_img, H_combined_mask, D_combined_mask, \
					cropped_image, stain_mask = self.crop_image(input_image, mask, model, mask_filename)

					# Predict mask
					if self.main_model != 'staining':
						# Save the generated mask as a numpy array
						np.savez_compressed(self.masks_ihc_cells_path + '/' + mask_filename + '.npz', mask=H_combined_mask)

						plt.imshow(H_combined_mask)  # Verwende 'gray' für Graustufen-Darstellung
						plt.title('H Combined Mask')
						plt.axis('off')  # Deaktiviere die Achsen, um das Bild besser darzustellen
						plt.show()

						#plt.show()

					# Speichere die letzte verarbeitete ID
					last_processed_id = mainID
					self.save_last_processed_id(last_processed_id, filename)


			except Exception as e:
				print(f"Error processing {name}: {e}")
				skipped_images.append((mainID, url))

		# Save skipped images to a text file
		if skipped_images:
			with open('skipped_images.txt', 'w') as f:
				for mainID, url in skipped_images:
					f.write(f"MainID: {mainID}, URL: {url}\n")

		# Save renamed files to a text file
		if renamed_files:
			with open('renamed_files.txt', 'w') as f:
				for mainID, url in renamed_files:
					f.write(f"MainID: {mainID}, URL: {url}\n")

			# # Further Analysis
			# D_props = regionprops_table(D_combined_mask, D_img,
			# 							properties=['label',
			# 										'area', 'equivalent_diameter',
			# 										'mean_intensity', 'solidity'])
			#
			# D_analysis_results = pd.DataFrame(D_props)
			# print(D_analysis_results.head())
			#
			# H_props = regionprops_table(H_combined_mask, H_img,
			# 							properties=['label',
			# 										'area', 'equivalent_diameter',
			# 										'mean_intensity', 'solidity'])
			#
			# H_analysis_results = pd.DataFrame(H_props)
			# print(H_analysis_results.head())
			#
			# D_mean_analysis_results = D_analysis_results.mean(axis=0)
			# D_mean_analysis_results
			#
			# H_mean_analysis_results = H_analysis_results.mean(axis=0)
			# H_mean_analysis_results
			#
			# D_total_area = D_analysis_results["area"].sum()
			# H_total_area = H_analysis_results["area"].sum()
			# positivity = D_total_area / (D_total_area + H_total_area)
			# print("The DAB positivity is: ", positivity)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	with open(r'environment/environment_variables.txt', 'r') as f:
		args.__dict__ = json.load(f)
	# General settings
	parser.add_argument("--database_path", default=os.getcwd() + args.database_path)
	parser.add_argument("--images_ihc_path", default=os.getcwd() + args.images_ihc_path)
	parser.add_argument("--masks_ihc_path", default=os.getcwd() + args.masks_ihc_path)
	parser.add_argument("--masks_ihc_cells_path", default=os.getcwd() + args.masks_ihc_cells_path)

	parser.add_argument("--model_path", default=os.getcwd() + args.model_path)

	parser.add_argument("--id_value", default=0, type=int, help="Select mainID (for example: 20000, 30000)")

	args = parser.parse_args()
	processing = cellpose(args)

	processing.main(args.id_value)