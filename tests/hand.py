
import cv2 as cv

img = cv.imread("images/hand.jpg")

img = cv.medianBlur(img, 7)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.Canny(img, 70, 120)

cv.imshow("test", img)

cv.waitKey(0)
cv.destroyAllWindows()
