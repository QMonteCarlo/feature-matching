#include "gms_matcher.h"

using namespace cv;
using namespace std;

size_t gms_matcher::GetInlierMask(vector<bool> &vbInliers, bool WithScale, bool WithRotation) {

    size_t max_inlier = 0;

    if (!WithScale && !WithRotation)
    {
        SetScale(0);
        max_inlier = run(1);
        vbInliers = mvbInlierMask;
        return max_inlier;
    }

    if (WithRotation && WithScale)
    {
        for (int Scale = 0; Scale < 5; Scale++)
        {
            SetScale(Scale);
            for (int RotationType = 1; RotationType <= 8; RotationType++)
            {
                size_t num_inlier = run(RotationType);

                if (num_inlier > max_inlier)
                {
                    vbInliers = mvbInlierMask;
                    max_inlier = num_inlier;
                }
            }
        }
        return max_inlier;
    }

    if (WithRotation && !WithScale)
    {
        for (int RotationType = 1; RotationType <= 8; RotationType++)
        {
            size_t num_inlier = run(RotationType);

            if (num_inlier > max_inlier)
            {
                vbInliers = mvbInlierMask;
                max_inlier = num_inlier;
            }
        }
        return max_inlier;
    }

    if (!WithRotation && WithScale)
    {
        for (int Scale = 0; Scale < 5; Scale++)
        {
            SetScale(Scale);

            size_t num_inlier = run(1);

            if (num_inlier > max_inlier)
            {
                vbInliers = mvbInlierMask;
                max_inlier = num_inlier;
            }

        }
        return max_inlier;
    }

    return max_inlier;
}



void gms_matcher::AssignMatchPairs(int GridType) {

    for (size_t i = 0; i < mNumberMatches; i++)
    {
        Point2f &lp = mvP1[mvMatches[i].first];
        Point2f &rp = mvP2[mvMatches[i].second];

        int lgidx = mvMatchPairs[i].first = GetGridIndexLeft(lp, GridType);
        int rgidx = -1;

        if (GridType == 1)
        {
            rgidx = mvMatchPairs[i].second = GetGridIndexRight(rp);
        }
        else
        {
            rgidx = mvMatchPairs[i].second;
        }

        if (lgidx < 0 || rgidx < 0)	continue;

        mMotionStatistics.at<int>(lgidx, rgidx)++;
        mNumberPointsInPerCellLeft[lgidx]++;
    }

}


void gms_matcher::VerifyCellPairs(int RotationType) {

    const int *CurrentRP = mRotationPatterns[RotationType - 1];

    for (int i = 0; i < mGridNumberLeft; i++)
    {
        if (sum(mMotionStatistics.row(i))[0] == 0)
        {
            mCellPairs[i] = -1;
            continue;
        }

        int max_number = 0;
        for (int j = 0; j < mGridNumberRight; j++)
        {
            int *value = mMotionStatistics.ptr<int>(i);
            if (value[j] > max_number)
            {
                mCellPairs[i] = j;
                max_number = value[j];
            }
        }

        int idx_grid_rt = mCellPairs[i];

        const int *NB9_lt = mGridNeighborLeft.ptr<int>(i);
        const int *NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt);

        int score = 0;
        double thresh = 0;
        int numpair = 0;

        for (size_t j = 0; j < 9; j++)
        {
            int ll = NB9_lt[j];
            int rr = NB9_rt[CurrentRP[j] - 1];
            if (ll == -1 || rr == -1)	continue;

            score += mMotionStatistics.at<int>(ll, rr);
            thresh += mNumberPointsInPerCellLeft[ll];
            numpair++;
        }

        thresh = THRESH_FACTOR * sqrt(thresh / numpair);

        if (score < thresh)
            mCellPairs[i] = -2;
    }

}

int gms_matcher::run(int RotationType) {

    mvbInlierMask.assign(mNumberMatches, false);

    // Initialize Motion Statisctics
    mMotionStatistics = Mat::zeros(mGridNumberLeft, mGridNumberRight, CV_32SC1);
    mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));

    for (int GridType = 1; GridType <= 4; GridType++)
    {
        // initialize
        mMotionStatistics.setTo(0);
        mCellPairs.assign(mGridNumberLeft, -1);
        mNumberPointsInPerCellLeft.assign(mGridNumberLeft, 0);

        AssignMatchPairs(GridType);
        VerifyCellPairs(RotationType);

        // Mark inliers
        for (size_t i = 0; i < mNumberMatches; i++)
        {
            if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
            {
                mvbInlierMask[i] = true;
            }
        }
    }
    int num_inlier = sum(mvbInlierMask)[0];
    return num_inlier;

}

//////////////////////////////////////////////////
size_t  gms_match(cv::Size size1, cv::Size size2,
                  std::vector<cv::KeyPoint> &vKpt1,
                  std::vector<cv::KeyPoint> &vKpt2,
                  std::vector<cv::DMatch> &matches)
{
//	cv::BFMatcher matcher(NORM_HAMMING);
//	matcher.match(Desc1, Desc2, matches);

    std::vector<bool> vbInliers;
    gms_matcher gms(vKpt1, size1, vKpt2, size2, matches);
    gms.GetInlierMask(vbInliers, false, false);

    size_t j = 0;
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches[j++] = matches[i];
        }
    }
    matches.resize(j);

    return matches.size();
}

