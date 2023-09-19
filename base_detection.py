import numpy as np
import cv2
import os

class BaseDetection():
    def __init__(self,
                 vertical_lines_offset = 320,
                 vertical_lines_nr = 1,
                 horizontal_lines_offset = 240,
                 horizontal_lines_nr = 1,
                 min_wall_length = 30):
        # DEFINE COLORS:
        self.BLACK = [0, 0, 0]
        self.BLUE = [255, 0, 0]
        self.GREEN = [0, 255, 0]
        self.RED = [0, 0, 255]
        self.YELLOW = [0, 255, 255]
        self.WHITE = [255, 255, 255]

        # min/max amount of pixels for detection
        self.min_wall_length = min_wall_length

        # line scans offset
        self.vertical_lines = []
        self.vertical_lines_nr = vertical_lines_nr
        self.arrangeVerticalLinesUniform(vertical_lines_offset, img_width=640)

        self.horizontal_lines = []
        self.horizontal_lines_nr = horizontal_lines_nr
        self.arrangeHorizontalLinesUniform(horizontal_lines_offset, img_height=480)

        self.mask_points = []
    
    def arrangeVerticalLinesUniform(self, vertical_lines_offset = 320, img_width = 640):
        vertical_lines = []
        for line_x in range(vertical_lines_offset, self.vertical_lines_nr*vertical_lines_offset+1, vertical_lines_offset):
            if line_x>5 and line_x<img_width-5: vertical_lines.append(line_x)
            else: print(f"Detection line out of resolution bounds! Vertical position:Line {line_x}")
        self.vertical_lines = vertical_lines

    def arrangeHorizontalLinesUniform(self, horizontal_lines_offset = 320, img_height = 480):
        horizontal_lines = []
        for line_y in range(horizontal_lines_offset, self.vertical_lines_nr*horizontal_lines_offset+1, horizontal_lines_offset):
            if line_y>5 and line_y<img_height-5: horizontal_lines.append(line_y)
            else: print(f"Detection line out of resolution bounds! Vertical position:Line {line_y}")
        self.horizontal_lines = horizontal_lines

    def arrangeVerticalLinesRandom(self, img_width = 640, detections = []):
        vertical_lines = []
        for i in range(self.vertical_lines_nr):
            line_x = int(np.random.uniform(5, img_width-5))
            while self.isInsideBoundingBox(line_x, detections): 
                line_x = int(np.random.uniform(5, img_width-5))
            vertical_lines.append(line_x)
        self.vertical_lines = vertical_lines

    def isInsideBoundingBox(self, x, detections):
        for detection in detections:
            class_id, score, xmin, xmax, ymin, ymax = detection
            if (x < xmax and x > xmin): 
                return True
        return False

    def updateMask(self, boundary_points):
        # TODO: make better approach for mask setting
        self.mask_points = boundary_points

    def isOutOfField(self, detection):
        # TODO: make better approach for out-of-field-checking
        return False

    def isBlack(self, src):
        blue, green, red = src
        if green < 70 and red < 70 and blue < 70:
            return True
        else:
            return False

    def isBlue(self, src):
        blue, green, red = src
        if blue > 69 and green < 130 and red < 80:
            return True
        else:
            return False
    def isYellow(self, src):
        blue, green, red = src
        if blue < 100 and green > 180 and red > 180: 
            return True
        else:
            return False

    def isGreen(self, src):
        blue, green, red = src
        if green > 90 and red <= 130:
            return True
        else:
            return False     

    def isWhite(self, src):
        blue, green, red = src
        if blue > 130 and green > 130 and red > 130:
            return True
        else:
            return False

    def segmentPixel(self, src):
        # CHANGE AVAILABLE COLORS
        if self.isYellow(src):
            color = self.YELLOW
            return color        
        elif self.isBlue(src):
            color = self.BLUE
            return color
        else:
            return src
    
    def segmentBase(self, src):
        """
        Make description here
        """
        # make copy from source image for segmentation
        # segmented_img = src.copy()
        segmented_img = src

        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        for line_x in self.vertical_lines:
            # segment vertical lines
            for pixel_y in range(0, height):
                pixel = src[pixel_y, line_x]
                color = self.segmentPixel(pixel)
                segmented_img[pixel_y, line_x] = color

        return segmented_img        
    
    def baseCircleDetection(self, src):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # circle detection points
        circle_boundary_points = []

        lower_point = self.baseLowerCircleDetection(src)
        upper_point = self.baseUpperCircleDetection(src)
        left_point = self.baseLeftCircleDetection(src)
        right_point = self.baseRightCircleDetection(src)

        circle_boundary_points.append(lower_point[0])
        circle_boundary_points.append(upper_point[0])
        circle_boundary_points.append(left_point[0])
        circle_boundary_points.append(right_point[0])

        return circle_boundary_points

    def circleBorderDetection(self, src):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # circle detection points
        circle_boundary_points = []

        for line_x in self.vertical_lines:
            circle_points = []
            for pixel_y in range(height-1, int(height/2), -1):
                pixel = src[pixel_y, line_x]
                pixel_is_blue_or_black = (self.isBlue(pixel) or self.isBlack(pixel))
                if len(circle_points)>self.min_wall_length and not pixel_is_blue_or_black:
                    circle_boundary_points.append(circle_points[-1])
                    break
                elif pixel_is_blue_or_black:
                    circle_points.append([pixel_y, line_x])
                else:
                    circle_points = []

        return circle_boundary_points        

    def baseLowerCircleDetection(self, src):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # circle detection points
        circle_boundary_points = []

        for line_x in self.vertical_lines:
            circle_points = []
            for pixel_y in range(height-1, int(height/2), -1):
                pixel = src[pixel_y, line_x]
                pixel_is_blue_or_black = (self.isBlue(pixel) or self.isBlack(pixel))
                if len(circle_points)>self.min_wall_length and not pixel_is_blue_or_black:
                    circle_boundary_points.append(circle_points[-1])
                    break
                elif pixel_is_blue_or_black:
                    circle_points.append([pixel_y, line_x])
                else:
                    circle_points = []

        return circle_boundary_points

    def baseUpperCircleDetection(self, src):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # circle detection points
        circle_boundary_points = []

        for line_x in self.vertical_lines:
            circle_points = []
            for pixel_y in range(1, int(height/2), 1):
                pixel = src[pixel_y, line_x]
                pixel_is_blue_or_black = (self.isBlue(pixel) or self.isBlack(pixel))
                if len(circle_points)>self.min_wall_length and not pixel_is_blue_or_black:
                    circle_boundary_points.append(circle_points[-1])
                    break
                elif pixel_is_blue_or_black:
                    circle_points.append([pixel_y, line_x])
                else:
                    circle_points = []

        return circle_boundary_points

    def baseLeftCircleDetection(self, src):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # circle detection points
        circle_boundary_points = []

        for line_y in self.horizontal_lines:
            circle_points = []
            for pixel_x in range(0, int(width/2), 1):
                pixel = src[line_y, pixel_x]
                pixel_is_blue_or_black = (self.isBlue(pixel) or self.isBlack(pixel))
                if len(circle_points)>self.min_wall_length and not pixel_is_blue_or_black:
                    circle_boundary_points.append(circle_points[-1])
                    break
                elif pixel_is_blue_or_black:
                    circle_points.append([line_y, pixel_x])
                else:
                    circle_points = []

        return circle_boundary_points   

    def baseRightCircleDetection(self, src):
        """
        Make descripition here
        """
        # height and width from image resolution
        height, width = src.shape[0], src.shape[1]

        # circle detection points
        circle_boundary_points = []

        for line_y in self.horizontal_lines:
            circle_points = []
            for pixel_x in range(width-1, int(width/2), -1):
                pixel = src[line_y, pixel_x]
                pixel_is_blue_or_black = (self.isBlue(pixel) or self.isBlack(pixel))
                if len(circle_points)>self.min_wall_length and not pixel_is_blue_or_black:
                    circle_boundary_points.append(circle_points[-1])
                    break
                elif pixel_is_blue_or_black:
                    circle_points.append([line_y, pixel_x])
                else:
                    circle_points = []

        return circle_boundary_points   

    def computeCircleFromTriangle(self, A, B, C):
        R90 = np.array([[0, -1], [1, 0]])
        lambda2_x = (B-A)@(A-C)
        lambda2_y = 2*(B-A)@R90@(C-B)
        if lambda2_y==0:
            lambda2 = 0
        else:
            lambda2 = lambda2_x/lambda2_y
        
        O = (B+C)/2 + lambda2*R90@(C-B)

        r = np.linalg.norm(O-A)

        return O, r

if __name__ == "__main__":
    cwd = os.getcwd()

    FRAME_NR = 5
    QUADRADO = 1
    WINDOW_NAME = "BOUNDARY DETECTION"
    VERTICAL_LINES_NR = 1

    # BASE DETECTION TESTS
    base_detector = BaseDetection(
                    vertical_lines_offset=320,
                    vertical_lines_nr=1,
                    min_wall_length=10)
    print(base_detector.vertical_lines)

    while True:
        IMG_PATH = cwd + f"/bases_pics/test_{FRAME_NR}.jpg"
        img = cv2.imread(IMG_PATH)

        circle_boundary_points = base_detector.baseCircleDetection(img)
        print(circle_boundary_points)
        
        A = np.array(circle_boundary_points[0])
        B = np.array(circle_boundary_points[1])
        C = np.array(circle_boundary_points[2])

        O, radius = base_detector.computeCircleFromTriangle(A, B, C)

        center_y, center_x = int(O[0]), int(O[1])
        cv2.drawMarker(img, (center_x, center_y), color=base_detector.RED)

        theta = np.deg2rad(45)
        radius_end = int(center_x+radius*np.cos(theta)), int(center_y+radius*np.sin(theta))
        cv2.line(img, (center_x, center_y), radius_end, color=base_detector.BLACK, thickness=2)

        for point in circle_boundary_points:
            pixel_y, pixel_x = point
            #img[pixel_y, pixel_x] = base_detector.RED
            cv2.drawMarker(img, (pixel_x, pixel_y), color=base_detector.RED)
            
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(-1) & 0xFF
        if key == ord('q'):
            break
        else:
            FRAME_NR += 1

    # RELEASE WINDOW AND DESTROY
    cv2.destroyAllWindows()