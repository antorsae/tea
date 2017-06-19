#include "didi_pipeline/tracklet.h"

using namespace std;


namespace didi
{

string indent(int tabCount, bool tabAsSpace = false)
{
	const int tabSpaces {4};

	string indentStr;
	if (tabAsSpace) {
		for (int i{}; i < tabCount * tabSpaces; ++i) indentStr.push_back(' ');
	}
	else {
		for (int i{}; i < tabCount; ++i) indentStr.push_back('\t');
	}

	return indentStr;
}

int Tracklet::writeXml(ofstream &f, int classId, int tabLevel, int firstFrameShift) const
{
	f << indent(tabLevel) << "<item class_id=\"" << classId << "\" tracking_level=\"0\" version=\"1\">" << endl;
	++tabLevel;
	++classId;
	f << indent(tabLevel) << "<objectType>" << objectType << "</objectType>" << endl;
	f << indent(tabLevel) << "<h>" << height << "</h>" << endl;
	f << indent(tabLevel) << "<w>" << width << "</w>" << endl;
	f << indent(tabLevel) << "<l>" << length << "</l>" << endl;
	f << indent(tabLevel) << "<first_frame>" << firstFrame + firstFrameShift << "</first_frame>" << endl;
	f << indent(tabLevel) << "<poses class_id=\"" << classId << "\" tracking_level=\"0\" version=\"0\">" << endl;
	++classId;
	++tabLevel;
	f << indent(tabLevel) << "<count>" << poses.size() << "</count>" << endl;
	f << indent(tabLevel) << "<item_version>2</item_version>" << endl;
	bool firstPose{true};
	for (const Pose &p : poses) {
		if (firstPose) {
			f << indent(tabLevel) << "<item class_id=\"" << classId << "\" tracking_level=\"0\" version=\"2\">" << endl;
			firstPose = false;
		}
		else {
			f << indent(tabLevel) << "<item>" << endl;
		}
		++tabLevel;
		++classId;
		f << indent(tabLevel) << "<tx>" << p.t.x() << "</tx>" << endl;
		f << indent(tabLevel) << "<ty>" << p.t.y() << "</ty>" << endl;
		f << indent(tabLevel) << "<tz>" << p.t.z() << "</tz>" << endl;
		f << indent(tabLevel) << "<tz>" << -height/2 << "</tz>" << endl;
		f << indent(tabLevel) << "<rx>" << p.r.x() << "</rx>" << endl;
		f << indent(tabLevel) << "<ry>" << p.r.y() << "</ry>" << endl;
		f << indent(tabLevel) << "<rz>" << p.r.z() << "</rz>" << endl;
		f << indent(tabLevel) << "<state>1</state>" << endl;
		f << indent(tabLevel) << "<occlusion>-1</occlusion>" << endl;
		f << indent(tabLevel) << "<occlusion_kf>-1</occlusion_kf>" << endl;
		f << indent(tabLevel) << "<truncation>-1</truncation>" << endl;
		f << indent(tabLevel) << "<amt_occlusion>0.0</amt_occlusion>" << endl;
		f << indent(tabLevel) << "<amt_occlusion_kf>-1</amt_occlusion_kf>" << endl;
		f << indent(tabLevel) << "<amt_border_l>0.0</amt_border_l>" << endl;
		f << indent(tabLevel) << "<amt_border_r>0.0</amt_border_r>" << endl;
		f << indent(tabLevel) << "<amt_border_kf>-1</amt_border_kf>" << endl;
		--tabLevel;
		f << indent(tabLevel) << "</item>" << endl;
	}
	--tabLevel;
	f << indent(tabLevel) << "</poses>" << endl;
	f << indent(tabLevel) << "<finished>1</finished>" << endl;
	--tabLevel;
	f << indent(tabLevel) << "</item>" << endl;

	return classId;
}

void writeTrackletsXml(string filename, const std::vector<Tracklet> &tlets, int firstFrameShift)
{
	ofstream f;
	f.open(filename.c_str());
	assert(f.is_open());

	int tabLevel{0};

	f << indent(tabLevel) << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>" << endl;
	f << indent(tabLevel) << "<!DOCTYPE boost_serialization>" << endl;
	f << indent(tabLevel) << "<boost_serialization signature=\"serialization::archive\" version=\"9\">" << endl;
	f << indent(tabLevel) << "<tracklets class_id=\"0\" tracking_level=\"0\" version=\"0\">" << endl;
	++tabLevel;
	f << indent(tabLevel) << "<count>" << tlets.size() << "</count>" << endl;
	f << indent(tabLevel) << "<item_version>1</item_version> " << endl;
	int classId{1};
	for (const Tracklet &tlet : tlets) {
		classId = tlet.writeXml(f, classId, tabLevel, firstFrameShift);
	}
	--tabLevel;
	f << indent(tabLevel) << "</tracklets>" << endl;
	f << indent(tabLevel) << "</boost_serialization> " << endl;
	f.close();
}


};
