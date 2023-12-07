#include <pybind11/stl.h>

class ByteTrie : public std::enable_shared_from_this<ByteTrie> { // enable_shared_from_this allows use to return a raw pointer to the parent
private:
    ByteTrie* _parent = nullptr; // Raw pointer since we are not owning the parent

public:
    int match_version = -1;
    bool match = false;
    bool partial_match = false;
    double prob = 0;
    int value = -1;
    std::unordered_map<char, std::shared_ptr<ByteTrie>> children;

    ByteTrie(std::vector<std::string> byte_strings) {
        for (size_t i = 0; i < byte_strings.size(); ++i) {
            insert(byte_strings[i], 0);
        }
    }

    ByteTrie(std::vector<std::string> byte_strings, std::vector<int> values) {
        for (size_t i = 0; i < byte_strings.size(); ++i) {
            insert(byte_strings[i], values[i]);
        }
    }

    ByteTrie(ByteTrie* parent) : _parent(parent) {}

    std::vector<char> keys() const {
        std::vector<char> keys;
        for (const auto& pair : children) {
            keys.push_back(pair.first);
        }
        return keys;
    }

    bool has_child(char byte) {
        return children.count(byte) > 0;
    }

    std::shared_ptr<ByteTrie> child(char byte) {
        return children[byte];
    }

    ByteTrie *parent() {
        return this->_parent;
    }

    size_t size() {
        return children.size();
    }

    void insert(const std::string& s, int value, unsigned int pos = 0) {
        if (s.size() <= pos) {
            if (this->value < 0) {
                this->value = value;
            }
        } else {
            uint8_t first_byte = s[pos];
            if (children.find(first_byte) == children.end()) {
                children[first_byte] = std::make_shared<ByteTrie>(this);
            }
            children[first_byte]->insert(s, value, pos + 1);
        }
    }

    // we could save a lot of work if we assume the top node has prob 1.0 and then only explore the subtree we care about
    void compute_probs(const std::vector<double>& probs) {
        prob = 0.0;
        
        if (value != -1) {
            prob += probs[value];
        }

        if (!children.empty()) {
            for (auto& pair : children) {
                pair.second->compute_probs(probs);
                prob += pair.second->prob;
            }
        }
    }
};
