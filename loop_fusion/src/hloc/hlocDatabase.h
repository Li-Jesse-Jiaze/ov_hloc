//
// Created by 96320 on 2022/4/16.
//

#ifndef TEST_HLOCDATABASE_H
#define TEST_HLOCDATABASE_H

#include <opencv2/core.hpp>
#include <map>

using namespace std;

/// Id of entries of the database
typedef int EntryId;

class Result {
public:

    EntryId Id{};
    double Score{};

    /**
     * Empty constructors
     */
    inline Result() = default;

    /**
     * Creates a result with the given data
     * @param _id entry id
     * @param _score score
     */
    inline Result(EntryId _id, double _score) : Id(_id), Score(_score) {}

    /**
     * Compares the scores of two results
     * @return true iff this.score < r.score
     */
    inline bool operator<(const Result &r) const {
        return this->Score < r.Score;
    }

    /**
     * Compares the scores of two results
     * @return true iff this.score > r.score
     */
    inline bool operator>(const Result &r) const {
        return this->Score > r.Score;
    }

    /**
     * Compares the entry id of the result
     * @return true iff this.id == id
     */
    inline bool operator==(EntryId id) const {
        return this->Id == id;
    }

    /**
     * Compares the score of this entry with a given one
     * @param s score to compare with
     * @return true iff this score < s
     */
    inline bool operator<(double s) const {
        return this->Score < s;
    }

    /**
     * Compares the score of this entry with a given one
     * @param s score to compare with
     * @return true iff this score > s
     */
    inline bool operator>(double s) const {
        return this->Score > s;
    }

    /**
     * Compares the score of two results
     * @param a
     * @param b
     * @return true iff a.Score > b.Score
     */
    static inline bool gt(const Result &a, const Result &b) {
        return a.Score > b.Score;
    }

    /**
     * Compares the scores of two results
     * @return true iff a.Score > b.Score
     */
    inline static bool ge(const Result &a, const Result &b) {
        return a.Score > b.Score;
    }

    /**
     * Returns true iff a.Score >= b.Score
     * @param a
     * @param b
     * @return true iff a.Score >= b.Score
     */
    static inline bool geq(const Result &a, const Result &b) {
        return a.Score >= b.Score;
    }

    /**
     * Returns true iff a.Score >= s
     * @param a
     * @param s
     * @return true iff a.Score >= s
     */
    static inline bool geqv(const Result &a, double s) {
        return a.Score >= s;
    }


    /**
     * Returns true iff a.Id < b.Id
     * @param a
     * @param b
     * @return true iff a.Id < b.Id
     */
    static inline bool ltId(const Result &a, const Result &b) {
        return a.Id < b.Id;
    }
};

class hlocDatabase {
public:
    void add(cv::Mat &new_global_descriptor) {
        database[database.size()] = new_global_descriptor;
    }

    void query(cv::Mat &global_descriptor, vector<Result> &ret, EntryId start_id, EntryId end_id) {
        for (int i = start_id; i < end_id; i++) {
            ret.emplace_back(i, global_descriptor.dot(database[i]));
        }
        sort(ret.begin(), ret.end(), greater<>());
    }

private:
    map<EntryId, cv::Mat> database;
};


#endif //TEST_HLOCDATABASE_H
