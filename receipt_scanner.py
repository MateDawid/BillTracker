from imutils.perspective import four_point_transform
import pytesseract
import imutils
import cv2


class ReceiptScanner:
	def __init__(self, image_path):
		self.image_path = image_path
		self.original_image = self.get_original_image()
		self.resized_image = self.get_resized_image()
		self.ratio = self.calculate_ratio()
		self.edged_image = self.get_edged_image()
		self.countours = self.get_contours()
		self.contoured_image = self.get_contoured_image()
		self.centered_image = self.get_centered_image()
		self.uncolored_image = self.get_uncolored_image()
		self.receipt_text = self.get_receipt_text()
		self.tutorial_text = self.get_tutorial_text(False)

	def get_original_image(self):
		return cv2.imread(self.image_path)

	def get_resized_image(self):
		return imutils.resize(self.original_image.copy(), width=500)

	def calculate_ratio(self):
		return self.original_image.shape[1] / float(self.resized_image.shape[1])

	def get_edged_image(self):
		"""" Converts the image to grayscale, blur it slightly, and then applies edge detection """
		gray = cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
		return cv2.Canny(blurred, 75, 200)

	def get_contours(self):
		# find contours in the edge map and sort them by size in descending
		# order
		contours = cv2.findContours(self.edged_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		# initialize a contour that corresponds to the receipt outline
		receipt_contours = None
		# loop over the contours
		for contour in contours:
			# approximate the contour
			peri = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
			# if our approximated contour has four points, then we can
			# assume we have found the outline of the receipt
			if len(approx) == 4:
				return approx
		# if the receipt contour is empty then our script could not find the
		# outline and we should be notified
		if receipt_contours is None:
			raise Exception("Could not find receipt outline. Try debugging your edge detection and contour steps.")

	def get_contoured_image(self):
		return cv2.drawContours(self.resized_image.copy(), [self.countours], -1, (0, 255, 0), 2)

	def get_centered_image(self):
		return four_point_transform(self.original_image, self.countours.reshape(4, 2) * self.ratio)

	def get_uncolored_image(self):
		return cv2.cvtColor(self.centered_image, cv2.COLOR_BGR2RGB)

	def get_receipt_text(self):
		return pytesseract.image_to_string(self.uncolored_image, config="--psm 4")

	def print_receipt_text(self):
		print(self.receipt_text)

	def display_image(self):
		available_images = {k: v for k, v in self.__dict__.items() if k.endswith('_image')}
		print('Available images:')
		for i, image in enumerate(available_images, start=1):
			print(f'{i}. {image}')
		input_image = input('Select image from list above: ').strip().lower()
		try:
			if input_image == 'centered_image':
				selected_image = imutils.resize(available_images[input_image], width=500)
			else:
				selected_image = available_images[input_image]
		except KeyError:
			print('ERROR: Wrong image name passed. Try again.')
			self.display_image()
		cv2.imshow(input_image.title().replace('_', ' '), selected_image)
		cv2.waitKey(0)


if __name__ == '__main__':
	images = ['275622063_360645385756004_7149566824326966301_n.jpg', '275600487_566229767710183_4015717798290752651_n.jpg']
	images = ['275622063_360645385756004_7149566824326966301_n.jpg']
	for path in images:
		pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract\tesseract.exe"
		scanner = ReceiptScanner(path)
