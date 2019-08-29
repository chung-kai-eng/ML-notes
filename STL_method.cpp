#include<iostream>
#include<string>
#include<vector>
#include<queue>
#include<stack>
#include<set>
#include<map> 
// the method to use STL (vector, queue, stack, set, map)
//https://larry850806.github.io/2016/06/06/STL1/

// method of vector extended function 
// https://mropengate.blogspot.com/2015/07/cc-vector-stl.html
using namespace std;

class input{
	public:
		input(string cmd, string src, string dst, string ins_itm, string del_itm, int method){
			_cmd = cmd;
			_src = src;
			_dst = dst;
			_ins_itm = ins_itm;
			_del_itm = del_itm;
			_method = method;
		}
		void print_rlt();	
	private:
		string _cmd;
		string _src;
		string _dst;
		string _ins_itm;
		string _del_itm;
		int _method;
};
void input::print_rlt(){
	cout << _cmd << ' ' << _src <<endl;
	cout << _ins_itm << ' ' << _del_itm << endl;
	cout << _method << endl;
}
class Rectangle{
	public:
		// initialization same as using  _height = height;
		Rectangle(int height, int weight) : _height(height), _weight(weight) {}
		int area(){
			return _height*_weight;
		}
	private:
		int _xLow, _yLow, _height, _weight;
		//(xLow, yLow) is the position of the left corner of rectangle
};
void vec_method();
void queue_method();
void stack_method();
void set_method();
void map_method();

int main(){
	/*
	Rectangle a(2,3);
	cout << a.area()<<endl;
	input item("INSERT", "start", "terminal", "a", "b", 2);
	item.print_rlt();
	*/
	
	//vec_method();
	//queue_method();
	//stack_method();
	//set_method();
	//map_method();
	return 0;	
}


void queue_method(){
	queue<string> q;
	cout << "queue" << endl;
	cout << "Queue is not an array." << endl;
	cout << "Thus, we can't use index like vector." << endl;
	cout << "Instead, we need to use pop to gain the element of queue." << endl;
	q.push("Andy");
	q.push("Cheryl");
	q.push("loves");
	q.push("Eric");
	q.pop();
	cout << "print the first element of queue : " << q.front() << endl;
	cout << "print the last element of queue : " << q.back() << endl;
	while(q.size() != 0){
		cout << q.front() << ' ';
		q.pop();
	}	
}
void vec_method(){
	vector<int> vec;
	for(int i = 0; i < 5; i++){
		vec.push_back(i*10); //[0, 10, 20 ,30 ,40]
	}
	vec.pop_back();
	vec.pop_back();
	cout << "vector" << endl;
	for(int i = 0 ; i < vec.size(); i++){
			cout << vec[i] << ' ';
	}
}
void stack_method(){
	stack<string> s; 
	s.push("Cheryl");
	s.push("loves");
	s.push("Eric");
	for(int i = s.size(); i != 0; i--){
		cout << s.top() << ' ';
		s.pop();
	}
}
void set_method(){
	set<int> mySet;
    mySet.insert(20);   // mySet = {20}
    mySet.insert(10);   // mySet = {10, 20}
    mySet.insert(30);   // mySet = {10, 20, 30}

    cout << mySet.count(20) << endl;    // 存在 -> 1
    cout << mySet.count(100) << endl;   // 不存在 -> 0
	cout << "after erase : " << endl;
    mySet.erase(20);                    // mySet = {10, 30}
    cout << mySet.count(20) << endl;    // 0
}
void map_method(){
	map<string, int> m;     // 從 string 對應到 int
                        // 設定對應的值
    m["one"] = 1;       // "one" -> 1
    m["two"] = 2;       // "two" -> 2
    m["three"] = 3;     // "three" -> 3

    cout << m.count("two") << endl;     // 1 -> 有對應
    cout << m.count("ten") << endl;     // 0 -> 沒有對應
    cout << m["one"] << endl;       // 1
    cout << m["three"] << endl;     // 3
    cout << m["ten"] << endl;       // 0 (無對應值)

}





