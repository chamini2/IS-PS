#include <cassert>
#include <cstdio>

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <unordered_map>
using std::unordered_map;

#include <set>
using std::multiset;

using std::pair;
using std::make_pair;

#include "point_interface.hpp"
#include "instance_selection.hpp"

class BalloonPoint : public PointInterface<bool> {
public:

    BalloonPoint(bool class_label, vector<float> attributes) : 
                        PointInterface<bool> ( class_label, attributes_ ) {
    }

    // attributes_[0] == YELLOW/PURPLE (0/1)
    // attributes_[1] == SMALL/LARGE   (0/1)
    // attributes_[2] == DIP/STRETCH   (0/1)
    // attributes_[3] == ADULT/CHILD   (0/1)
    float distance(PointInterface<bool> obj) {
        int size = attributes_.size();
        float distance = 0;

        for (int i = 0; i < size; ++i) {
            if (attributes_[i] != obj.attributes()[i]) {
                ++distance;
            }
        }

        return distance;
    }

};

inline bool operator < (const BalloonPoint& lhs, const BalloonPoint& rhs) {
    int size = lhs.attributes().size();

    for (int i = 0; i < size; ++i) {
        if (lhs.attributes()[i] != rhs.attributes()[i]) {
            return lhs.attributes()[i] < rhs.attributes()[i];
        }
    }

    return false;
}

pair<bool, vector<float> > ParseCSV(unordered_map<string, float>& rule,
                                    unordered_map<string, bool>& class_rule, 
                                    char* line);
multiset< BalloonPoint > ParseFile(unordered_map<string, float>& rule, 
                                            unordered_map<string, bool>& class_rule, 
                                            const char* file);

int main(int argc, char const *argv[]) {

    multiset<BalloonPoint> points;

    unordered_map<string, float> rule = {
                                        {"YELLOW",0.0}, {"PURPLE",1.0}, 
                                        {"SMALL",0.0},  {"LARGE",1.0}, 
                                        {"DIP",0.0},    {"STRETCH",1.0}, 
                                        {"ADULT",0.0},  {"CHILD",1.0}
                                    };
    unordered_map<string, bool> class_rule = {{"T",true}, {"F",false}};

    printf("Reading file\n");
    fflush(stdout);
    points = ParseFile(rule, class_rule, argv[1]);
    printf("Reading file\n");

    return 0;
}

multiset<BalloonPoint> ParseFile(unordered_map<string, float>& rule, 
                                            unordered_map<string, bool>& class_rule, 
                                            const char* file) {
    FILE *fp;
    char *line = NULL;
    size_t len = 0;

    multiset< BalloonPoint > points;

    fp = fopen(file, "r");
    assert(fp != NULL);

    while (getline(&line, &len, fp) != -1) {
        printf("%s\n", line);
        fflush(stdout);

        auto balloon_pair = ParseCSV(rule, class_rule, line);

        printf("after\n");
        fflush(stdout);
        
        BalloonPoint bp = BalloonPoint(balloon_pair.first, balloon_pair.second);

        printf("after3\n");
        fflush(stdout);

        points.insert(bp);

        printf("after2\n");
        fflush(stdout);
    }

    return points;
}

pair<bool, vector<float> > ParseCSV(unordered_map<string, float>& rule,
                                    unordered_map<string, bool>& class_rule, 
                                    char* line) {
    char *next, *field;
    vector<float> attributes;

    field = strtok(line, ",");
    next = strtok(NULL, ",");
    while (next != NULL) {
        printf("%s", field);
        printf(" | %s\n", next);
        fflush(stdout);

        attributes.push_back(rule[field]);
        field = next;
        next = strtok(NULL, ",");
    }

    for (float f : attributes) {
        printf("%f, ", f);
    }
    printf("\n");

    field[strlen(field)-1] = '\0';
    printf("'%d'\n", class_rule[field]);
    fflush(stdout);

    return make_pair(class_rule[field], attributes);
}
