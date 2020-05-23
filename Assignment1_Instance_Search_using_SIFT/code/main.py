# import the three coding blocks
import data_process
import color_hist
import SIFT


def main():
    # Entry point of the program
    # Read images into the program
    write_to_file_path = None
    data_dir = "../../ug_data/Images"
    test_mode = True
    data = data_process.read_from_img(data_dir, write_to_file_path, test_mode)
    # data = data_process.read_from_file(write_to_file_path)

    query_dir = "../../ug_data/examples/example_query"
    query_data = data_process.read_from_img(query_dir)

    test_query_dir = "../../ug_data/Queries"
    test_query_data = data_process.read_from_img(test_query_dir)

    # The mode you want to apply:
    # Option 1: "Color_hist"
    # Option 2: "SIFT"
    mode = 'Color_hist'

    if mode == 'Color_hist':
        print("Using Color Histogram")
        # Mean Average Precision: 0.073896
        # Q3 & Q6 are good, Q3: 0.3658, Q6: 0.2611

        print("Start to convert all image data.")
        # Convert images in the database into color histograms
        data = color_hist.convert_data(data)

        # 10 example queries
        print("Start to process the example queries.")
        # Compute the similarity and return the ranking result
        color_hist_result = color_hist.color_hist(query_data, data)
        # Write the ranking result into the file
        data_process.write_result(color_hist_result, 'color_hist.txt')

        # 20 test queries
        print("Start to process the test queries.")
        test_color_hist_result = color_hist.color_hist(test_query_data, data)
        data_process.write_result(test_color_hist_result, 'test_color_hist.txt')

    elif mode == 'SIFT':
        print("Using SIFT")
        # Average Precision of Q1: 0.2509
        # Average Precision of Q2: 0.8836
        # Average Precision of Q3: 1.0000
        # Average Precision of Q4: 0.0302
        # Average Precision of Q5: 0.3958
        # Average Precision of Q6: 0.9488
        # Average Precision of Q7: 0.0050
        # Average Precision of Q8: 0.4727
        # Average Precision of Q9: 0.0254
        # Average Precision of Q10: 0.2602
        # Mean Average Precision: 0.427257

        print("Start to convert all image data.")
        # Detect the corners of the images in the database, and compute the SIFT feature
        data = SIFT.detect_data(data)

        # 10 example queries
        print("Start to process the example queries.")
        # Compute the similarity and return the ranking result
        max_result = SIFT.SIFT_perspective_change(query_data, data)
        # Write the ranking result into the file
        data_process.write_result(max_result, 'max_sift.txt')

        # 20 test queries
        print("Start to process the test queries.")
        test_max_result = SIFT.SIFT_perspective_change(test_query_data, data)
        data_process.write_result(test_max_result, 'test_max_sift.txt')


if __name__ == "__main__":
    main()
