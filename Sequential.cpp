#include <iostream>
#include <fstream>
#include <climits>
#include <cstring>
#include <map>
#include <string>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <queue>
#include <set>

using namespace std;

const int MAX_NODES = 10000;
const int INF = INT_MAX;

map<int, int> id_to_index;
int index_to_id[MAX_NODES];
int currentIndex = 0;

struct AdjNode {
    int dest;
    int weight;
    AdjNode* next;
};

struct Graph {
    int V;
    AdjNode* adj[MAX_NODES];

    Graph(int vertices) {
        V = vertices;
        for (int i = 0; i < V; ++i)
            adj[i] = nullptr;
    }

    void addEdge(int src, int dest, int weight) {
        removeEdge(src, dest);
        AdjNode* newNode = new AdjNode{dest, weight, adj[src]};
        adj[src] = newNode;
    }

    void removeEdge(int src, int dest) {
        AdjNode* curr = adj[src];
        AdjNode* prev = nullptr;
        while (curr) {
            if (curr->dest == dest) {
                if (prev) prev->next = curr->next;
                else adj[src] = curr->next;
                delete curr;
                break;
            }
            prev = curr;
            curr = curr->next;
        }
    }

    void clearGraph() {
        for (int i = 0; i < V; ++i) {
            AdjNode* curr = adj[i];
            while (curr) {
                AdjNode* next = curr->next;
                delete curr;
                curr = next;
            }
            adj[i] = nullptr;
        }
    }
};

void incrementalSSSP(Graph& g, int dist[], int parent[], set<int>& affected) {
    set<int> worklist = affected;
    while (!worklist.empty()) {
        set<int> next_worklist;
        for (int u : worklist) {
            AdjNode* curr = g.adj[u];
            while (curr) {
                int v = curr->dest;
                int weight = curr->weight;
                if (dist[u] != INF && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    parent[v] = u;
                    next_worklist.insert(v);
                }
                curr = curr->next;
            }
        }
        worklist = next_worklist;
    }
}

void printPath(int node, int parent[], ostream& out) {
    if (node == -1) return;
    printPath(parent[node], parent, out);
    out << index_to_id[node] << " ";
}

void printAndSaveDistances(int dist[], int parent[], int V, int src, double execution_time) {
    ofstream fout("output.txt");
    cout << "\nSSSP from node " << index_to_id[src] << ":\n";
    fout << "SSSP from node " << index_to_id[src] << ":\n";

    for (int i = 0; i < V; ++i) {
        if (dist[i] == INF) {
            cout << "Node " << index_to_id[i] << ": Unreachable\n";
            fout << "Node " << index_to_id[i] << ": Unreachable\n";
        } else {
            cout << "Node " << index_to_id[i] << ": Distance = " << dist[i] << ", Path = ";
            fout << "Node " << index_to_id[i] << ": Distance = " << dist[i] << ", Path = ";
            printPath(i, parent, cout);
            printPath(i, parent, fout);
            cout << "\n";
            fout << "\n";
        }
    }

    cout << "Execution Time: " << execution_time << " seconds\n";
    fout << "Execution Time: " << execution_time << " seconds\n";
    fout.close();
}

int main() {
    srand(time(0));
    ifstream fin("Wiki-Vote.txt");
    if (!fin) {
        cout << "Error opening Wiki-Vote.txt\n";
        return 1;
    }

    int u_id, v_id;
    Graph g(MAX_NODES);
    string line;

    while (getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        sscanf(line.c_str(), "%d%d", &u_id, &v_id);

        if (id_to_index.find(u_id) == id_to_index.end()) {
            id_to_index[u_id] = currentIndex;
            index_to_id[currentIndex] = u_id;
            currentIndex++;
        }
        if (id_to_index.find(v_id) == id_to_index.end()) {
            id_to_index[v_id] = currentIndex;
            index_to_id[currentIndex] = v_id;
            currentIndex++;
        }

        int u = id_to_index[u_id];
        int v = id_to_index[v_id];
        g.addEdge(u, v, 1);
    }
    fin.close();

    int src_id, src;
    while (true) {
        cout << "Enter source node ID for SSSP (-1 to quit): ";
        cin >> src_id;

        if (src_id == -1) {
            cout << "Exiting...\n";
            return 0;
        }

        if (id_to_index.find(src_id) != id_to_index.end()) {
            src = id_to_index[src_id];
            break;
        } else {
            cout << "❌ Source node ID not found in the graph. Try again.\n";
        }
    }

    int dist[MAX_NODES], parent[MAX_NODES];

    for (int i = 0; i < MAX_NODES; ++i) {
        dist[i] = INF;
        parent[i] = -1;
    }
    dist[src] = 0;

    set<int> affected;
    affected.insert(src); // Initial worklist from source
    incrementalSSSP(g, dist, parent, affected);

    int total_updates;
    cout << "\nEnter total number of updates to perform: ";
    cin >> total_updates;

    affected.clear();

    for (int i = 0; i < total_updates; ++i) {
        bool is_insertion = rand() % 2;
        int from_id = rand() % 10000 + 1;
        int to_id = rand() % 10000 + 1;

        if (id_to_index.find(from_id) == id_to_index.end()) {
            id_to_index[from_id] = currentIndex;
            index_to_id[currentIndex] = from_id;
            currentIndex++;
        }
        if (id_to_index.find(to_id) == id_to_index.end()) {
            id_to_index[to_id] = currentIndex;
            index_to_id[currentIndex] = to_id;
            currentIndex++;
        }

        int u = id_to_index[from_id];
        int v = id_to_index[to_id];

        if (is_insertion) {
            cout << "\n[Update " << (i+1) << "] Inserting edge " << from_id << " -> " << to_id << "\n";
            g.addEdge(u, v, 1);
        } else {
            cout << "\n[Update " << (i+1) << "] Deleting edge " << from_id << " -> " << to_id << "\n";
            g.removeEdge(u, v);
        }

        if (dist[u] != INF) affected.insert(u); // Only affected if reachable
    }

    auto start = chrono::high_resolution_clock::now();
    incrementalSSSP(g, dist, parent, affected);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    printAndSaveDistances(dist, parent, currentIndex, src, duration.count());

    g.clearGraph();
    return 0;
}
