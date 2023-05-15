## This class is to retrieve the most important features with its conditions in descending order
if __name__ == "__main__":
    file = 'test.txt'
    file1 = open(file, 'r')
    lines = file1.readlines()
    final_dict = dict()

    totalinfluence = 0

    # Enter in all the feature conditions from the LSTM output into test.txt file
    for l in lines:
        mk1 = l.find("('") + 2
        mk2 = l.find("',", mk1)
        condition = l[mk1:mk2]
        influence = float(l.split(", ")[1].split(")")[0])
        # Sum of all the features for reference when doing influence percentage
        totalinfluence += influence

        # For repeating feature conditions, add influence as a sum
        if condition in final_dict:
            final_dict[condition]["count"] = final_dict[condition]["count"] + 1
            final_dict[condition]["influence"] = final_dict[condition]["influence"] + influence
            pass
        else:
            final_dict[condition] = {"count":1, "influence":influence}

    # Prints out the feature conditions in descending order regarding influence score
    print(dict(sorted(final_dict.items(), key=lambda item: item[1]["influence"], reverse=True)))
    # Prints out sum of influence score of all features
    print(totalinfluence)