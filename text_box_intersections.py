def check_all_intersections(bboxes):
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            a = bboxes[i]
            b = bboxes[j]
            a1, a2, a3, a4 = a
            b1, b2, b3, b4 = b

            if (b1[x] < a4[x] & b1[y] < a4[y]):
                x=x 
            if (b2[y] < a4[y] & b2[x] > a4[x]):
                x=x
            


            intersection = x
            if intersection == "vertical":
                new_a, new_b = split_vertically(a, b)
            elif intersection == "horizontal":
                new_a, new_b = split_horizontally(a, b)
            else:
                continue  # No intersection or not a supported intersection type

            bboxes[i] = new_a
            bboxes[j] = new_b

    return bboxes

def is_intersecting(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b

    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

def split_horizontally(a, b):

    return new_a, new_b

def split_vertically(a, b):
    
    return new_a, new_b
