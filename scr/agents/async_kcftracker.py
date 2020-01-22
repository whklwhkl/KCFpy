import numpy as np
import cv2


# ffttools
async def fftd(img, backwards=False):
	# shape of img can be (m,n), (m,n,1) or (m,n,2)
	# in my test, fft provided by numpy and scipy are slower than cv2.dft
	return cv2.dft(np.float32(img), flags = ((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))   # 'flags =' is necessary!

async def real(img):
	return img[:,:,0]

async def imag(img):
	return img[:,:,1]

async def complexMultiplication(a, b):
	res = np.zeros(a.shape, a.dtype)

	res[:,:,0] = a[:,:,0]*b[:,:,0] - a[:,:,1]*b[:,:,1]
	res[:,:,1] = a[:,:,0]*b[:,:,1] + a[:,:,1]*b[:,:,0]
	return res

async def complexDivision(a, b):
	res = np.zeros(a.shape, a.dtype)
	divisor = 1. / (b[:,:,0]**2 + b[:,:,1]**2)

	res[:,:,0] = (a[:,:,0]*b[:,:,0] + a[:,:,1]*b[:,:,1]) * divisor
	res[:,:,1] = (a[:,:,1]*b[:,:,0] + a[:,:,0]*b[:,:,1]) * divisor
	return res

async def rearrange(img):
	#return np.fft.fftshift(img, axes=(0,1))
	assert(img.ndim==2)
	img_ = np.zeros(img.shape, img.dtype)
	xh, yh = img.shape[1]//2, img.shape[0]//2
	img_[0:yh,0:xh], img_[yh:img.shape[0],xh:img.shape[1]] = img[yh:img.shape[0],xh:img.shape[1]], img[0:yh,0:xh]
	img_[0:yh,xh:img.shape[1]], img_[yh:img.shape[0],0:xh] = img[yh:img.shape[0],0:xh], img[0:yh,xh:img.shape[1]]
	return img_

# recttools
async def x2(rect):
	return rect[0] + rect[2]

async def y2(rect):
	return rect[1] + rect[3]

async def limit(rect, limit):
	if(rect[0]+rect[2] > limit[0]+limit[2]):
		rect[2] = limit[0]+limit[2]-rect[0]
	if(rect[1]+rect[3] > limit[1]+limit[3]):
		rect[3] = limit[1]+limit[3]-rect[1]
	if(rect[0] < limit[0]):
		rect[2] -= (limit[0]-rect[0])
		rect[0] = limit[0]
	if(rect[1] < limit[1]):
		rect[3] -= (limit[1]-rect[1])
		rect[1] = limit[1]
	if(rect[2] < 0):
		rect[2] = 0
	if(rect[3] < 0):
		rect[3] = 0
	return rect

async def getBorder(original, limited):
	res = [0,0,0,0]
	res[0] = limited[0] - original[0]
	res[1] = limited[1] - original[1]
	res[2] = await x2(original) - await x2(limited)
	res[3] = await y2(original) - await y2(limited)
	assert(np.all(np.array(res) >= 0))
	return res

async def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
	cutWindow = [x for x in window]
	await limit(cutWindow, [0,0,img.shape[1],img.shape[0]])   # modify cutWindow
	assert(cutWindow[2]>=0 and cutWindow[3]>=0)
	border = await getBorder(window, cutWindow)
	res = img[cutWindow[1]:cutWindow[1]+cutWindow[3], cutWindow[0]:cutWindow[0]+cutWindow[2]]

	if(border != [0,0,0,0]):
		res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
	return res

async def subPixelPeak(self, left, center, right):
	divisor = 2*center - right - left   #float
	return (0 if abs(divisor)<1e-3 else 0.5*(right-left)/divisor)

async def createGaussianPeak(self, sizey, sizex):
	syh, sxh = sizey/2, sizex/2
	output_sigma = np.sqrt(sizex*sizey) / self.padding * self.output_sigma_factor
	mult = -0.5 / (output_sigma*output_sigma)
	y, x = np.ogrid[0:sizey, 0:sizex]
	y, x = (y-syh)**2, (x-sxh)**2
	res = np.exp(mult * (y+x))
	return await fftd(res)

async def gaussianCorrelation(self, x1, x2):
	c = cv2.mulSpectrums(await fftd(x1), await fftd(x2), 0, conjB = True)   # 'conjB=' is necessary!
	c = await fftd(c, True)
	c = await real(c)
	c = await rearrange(c)

	if(x1.ndim==3 and x2.ndim==3):
		d = (np.sum(x1[:,:,0]*x1[:,:,0]) + np.sum(x2[:,:,0]*x2[:,:,0]) - 2.0*c) / (self.size_patch[0]*self.size_patch[1]*self.size_patch[2])
	elif(x1.ndim==2 and x2.ndim==2):
		d = (np.sum(x1*x1) + np.sum(x2*x2) - 2.0*c) / (self.size_patch[0]*self.size_patch[1]*self.size_patch[2])

	d = d * (d>=0)
	d = np.exp(-d / (self.sigma*self.sigma))

	return d

async def getFeatures(self, image, inithann, scale_adjust=1.0):
	extracted_roi = [0,0,0,0]   #[int,int,int,int]
	cx = self._roi[0] + self._roi[2]/2  #float
	cy = self._roi[1] + self._roi[3]/2  #float

	if(inithann):
		padded_w = self._roi[2] * self.padding
		padded_h = self._roi[3] * self.padding

		if(self.template_size > 1):
			if(padded_w >= padded_h):
				self._scale = padded_w / float(self.template_size)
			else:
				self._scale = padded_h / float(self.template_size)
			self._tmpl_sz[0] = int(padded_w // self._scale)
			self._tmpl_sz[1] = int(padded_h // self._scale)
		else:
			self._tmpl_sz[0] = int(padded_w)
			self._tmpl_sz[1] = int(padded_h)
			self._scale = 1.

		self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
		self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

	extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0])
	extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
	extracted_roi[0] = int(cx - extracted_roi[2]/2)
	extracted_roi[1] = int(cy - extracted_roi[3]/2)

	z = await subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
	if(z.shape[1]!=self._tmpl_sz[0] or z.shape[0]!=self._tmpl_sz[1]):
		z = cv2.resize(z, tuple(self._tmpl_sz))

	if(z.ndim==3 and z.shape[2]==3):
		FeaturesMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)   # z:(size_patch[0], size_patch[1], 3)  FeaturesMap:(size_patch[0], size_patch[1])   #np.int8  #0~255
	elif(z.ndim==2):
		FeaturesMap = z   #(size_patch[0], size_patch[1]) #np.int8  #0~255
	FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
	self.size_patch = [z.shape[0], z.shape[1], 1]

	# if(inithann):
	# 	self.createHanningMats()  # createHanningMats need size_patch
	#
	# FeaturesMap = self.hann * FeaturesMap
	return FeaturesMap

async def detect(self, z, x):
	k = await gaussianCorrelation(self, x, z)
	res = await real(await fftd(await complexMultiplication(self._alphaf, await fftd(k)), True))

	_, pv, _, pi = cv2.minMaxLoc(res)   # pv:float  pi:tuple of int
	p = [float(pi[0]), float(pi[1])]   # cv::Point2f, [x,y]  #[float,float]

	if(pi[0]>0 and pi[0]<res.shape[1]-1):
		p[0] += await subPixelPeak(self, res[pi[1],pi[0]-1], pv, res[pi[1],pi[0]+1])
	if(pi[1]>0 and pi[1]<res.shape[0]-1):
		p[1] += await subPixelPeak(self, res[pi[1]-1,pi[0]], pv, res[pi[1]+1,pi[0]])

	p[0] -= res.shape[1] / 2.
	p[1] -= res.shape[0] / 2.

	return p, pv

# side effect
async def train(self, x, train_interp_factor):
	k = await gaussianCorrelation(self, x, x)
	alphaf = await complexDivision(self._prob, await fftd(k) + self.lambdar)

	self._tmpl = (1-train_interp_factor)*self._tmpl + train_interp_factor*x
	self._alphaf = (1-train_interp_factor)*self._alphaf + train_interp_factor*alphaf
	return self._roi

# side effect
async def update(self, image):
	if(self._roi[0]+self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 1
	if(self._roi[1]+self._roi[3] <= 0):  self._roi[1] = -self._roi[2] + 1
	if(self._roi[0] >= image.shape[1]-1):  self._roi[0] = image.shape[1] - 2
	if(self._roi[1] >= image.shape[0]-1):  self._roi[1] = image.shape[0] - 2

	cx = self._roi[0] + self._roi[2]/2.
	cy = self._roi[1] + self._roi[3]/2.

	loc, peak_value = await detect(self, self._tmpl, await getFeatures(self, image, 0, 1.0))

	if(self.scale_step != 1):
		# Test at a smaller _scale
		new_loc1, new_peak_value1 = await detect(self, self._tmpl, await getFeatures(self, image, 0, 1.0/self.scale_step))
		# Test at a bigger _scale
		new_loc2, new_peak_value2 = await detect(self, self._tmpl, await getFeatures(self, image, 0, self.scale_step))

		if(self.scale_weight*new_peak_value1 > peak_value and new_peak_value1>new_peak_value2):
			loc = new_loc1
			peak_value = new_peak_value1
			self._scale /= self.scale_step
			self._roi[2] /= self.scale_step
			self._roi[3] /= self.scale_step
		elif(self.scale_weight*new_peak_value2 > peak_value):
			loc = new_loc2
			peak_value = new_peak_value2
			self._scale *= self.scale_step
			self._roi[2] *= self.scale_step
			self._roi[3] *= self.scale_step

	self._roi[0] = cx - self._roi[2]/2.0 + loc[0]*self.cell_size*self._scale
	self._roi[1] = cy - self._roi[3]/2.0 + loc[1]*self.cell_size*self._scale

	if(self._roi[0] >= image.shape[1]-1):  self._roi[0] = image.shape[1] - 1
	if(self._roi[1] >= image.shape[0]-1):  self._roi[1] = image.shape[0] - 1
	if(self._roi[0]+self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 2
	if(self._roi[1]+self._roi[3] <= 0):  self._roi[1] = -self._roi[3] + 2
	assert(self._roi[2]>0 and self._roi[3]>0)

	x = await getFeatures(self, image, 0, 1.0)
	return await train(self, x, self.interp_factor)
