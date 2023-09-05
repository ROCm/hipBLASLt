/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdint>
#include <stdexcept>
#include <Tensile/Tensile.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Serialization.hpp>
#include <Tensile/msgpack/MessagePack.hpp>
#include <msgpack.hpp>
#include <libgen.h>
#include "rocblaslt-auxiliary.h"

class SoftmaxProblem;
class SoftmaxSolution;

class SoftmaxSolution : public Tensile::Solution {
public:
    friend struct Tensile::Serialization::MappingTraits<SoftmaxSolution, Tensile::Serialization::MessagePackInput>;

    using Problem = SoftmaxProblem;
    std::string name() const override {
        return kernelName;
    }

    std::string description() const override {
        std::stringstream ss;
        ss << "Softmax, (Datatype, tileM, tileN) = "
           << "("
           << Tensile::ToString(datatype)
           << ", "
           << tileM
           << ", "
           << tileN
           << ")";
        return ss.str();
    }

    std::uint32_t getTileM() const {
        return tileM;
    }

    std::uint32_t getTileN() const {
        return tileN;
    }

    std::uint32_t getNumWorkitems() const {
        return numWorkitems;
    }

    std::string getCodeObjectPath() const {
        return coPath;
    }

    Tensile::DataType getDatatype() const {
        return datatype;
    }

private:
    std::size_t tileM{};
    std::size_t tileN{};
    std::size_t numWorkitems{};
    std::string coPath;
    std::string kernelName;
    Tensile::DataType datatype;
};

template<typename IO>
struct Tensile::Serialization::MappingTraits<SoftmaxSolution, IO>
{
    using iot = IOTraits<IO>;
    static void mapping(IO& io, SoftmaxSolution& s)
    {
        iot::mapRequired(io, "func_name", s.kernelName);
        std::string datatypeStr;
        iot::mapRequired(io, "io_type", datatypeStr);

        if (datatypeStr == "S") {
            s.datatype = Tensile::DataType::Float;
        } else {
            throw std::runtime_error("Invalid datatype in ext op library");
        }

        iot::mapRequired(io, "num_rows", s.tileM);
        iot::mapRequired(io, "num_cols", s.tileN);
        iot::mapRequired(io, "num_workitems", s.numWorkitems);
        iot::mapRequired(io, "co_path", s.coPath);
    }

    const static bool flow = false;
};

class SoftmaxProblem : public Tensile::Problem {
public:
    using Solution = SoftmaxSolution;
    SoftmaxProblem(uint32_t m, uint32_t n, Tensile::DataType datatype)
    : m(m), n(n), datatype(datatype) {}

    ~SoftmaxProblem() override {}

    std::string description() const override {
        std::stringstream ss;
        ss << "Softmax Problem(" << ToString(datatype) << ", " << m << ", " << n << ")";
        return ss.str();
    }

    std::uint32_t getM() const {
        return m;
    }

    std::uint32_t getN() const {
        return n;
    }
private:
    std::uint32_t m{};
    std::uint32_t n{};
    Tensile::DataType datatype{Tensile::DataType::Float};
};

struct ExtOpLibrary {
    virtual ~ExtOpLibrary() = default;
    virtual std::string type() const = 0;
    virtual std::string description() const = 0;

    template<typename T> 
    T &as() {
        return dynamic_cast<T &>(*this);
    }

    template<typename T> 
    const T &as() const {
        return dynamic_cast<T &>(*this);
    }
};

class SoftmaxSolutionLibrary : public ExtOpLibrary {
public:
    static constexpr char opName[] = "Softmax";

    ~SoftmaxSolutionLibrary() override {}
    void addSolution(SoftmaxSolution &sol) {
        solutions.push_back(std::make_shared<SoftmaxSolution>(sol));
    }

    std::string type() const override {
        return "SoftmaxSolutionLibrary";
    }

    std::string description() const override {
        return "SoftmaxSolutionLibrary";
    }

    std::shared_ptr<SoftmaxSolution> findBestSolution(
        const SoftmaxProblem &problem,
        const Tensile::Hardware &hardware,
        double* fitness = nullptr) const {
        auto bestSolIter = std::lower_bound(begin(solutions), end(solutions), problem.getN(), [](const auto &it, auto v) {
            return it->getTileN() < v;
        });

        return *bestSolIter;
    }

    void sortSolutions() {
        std::sort(begin(solutions), end(solutions), [](const auto &lhs, const auto &rhs) {
            return lhs->getTileN() < rhs->getTileN();
        });
    }

private:
    Tensile::SolutionVector<SoftmaxSolution> solutions;
};


class LayerNormProblem;
class LayerNormSolution;

class LayerNormSolution : public Tensile::Solution {
public:
    friend struct Tensile::Serialization::MappingTraits<LayerNormSolution, Tensile::Serialization::MessagePackInput>;

    using Problem = LayerNormProblem;
    std::string name() const override {
        return kernelName;
    }

    std::string description() const override {
        std::stringstream ss;
        ss << "LayerNorm, (Datatype) = "
           << "("
           << Tensile::ToString(datatype)
           << ")";
        return ss.str();
    }

    std::uint32_t getNumWorkitems() const {
        return numWorkitems;
    }

    std::uint32_t getLimit() const {
        return limit;
    }

    std::string getCodeObjectPath() const {
        return coPath;
    }

    Tensile::DataType getDatatype() const {
        return datatype;
    }

private:
    std::size_t numWorkitems{};
    std::size_t limit;
    std::string coPath;
    std::string kernelName;
    Tensile::DataType datatype;
};

template<typename IO>
struct Tensile::Serialization::MappingTraits<LayerNormSolution, IO>
{
    using iot = IOTraits<IO>;
    static void mapping(IO& io, LayerNormSolution& s)
    {
        std::string datatypeStr;

        // add co_path, remove arch and op
        iot::mapRequired(io, "co_path", s.coPath);
        iot::mapRequired(io, "func_name", s.kernelName);
        iot::mapRequired(io, "io_type", datatypeStr);
        iot::mapRequired(io, "num_workitems", s.numWorkitems);
        iot::mapRequired(io, "limit", s.limit);

        if (datatypeStr == "S") {
            s.datatype = Tensile::DataType::Float;
        } else {
            throw std::runtime_error("Invalid datatype in ext op library");
        }

    }

    const static bool flow = false;
};

class LayerNormProblem : public Tensile::Problem {
public:
    using Solution = LayerNormSolution;
    LayerNormProblem(uint32_t m, uint32_t n, Tensile::DataType datatype)
    : m(m), n(n), datatype(datatype) {}

    ~LayerNormProblem() override {}

    std::string description() const override {
        std::stringstream ss;
        ss << "LayerNorm Problem(" << ToString(datatype) << ", " << m << ", " << n << ")";
        return ss.str();
    }

    std::uint32_t getM() const {
        return m;
    }

    std::uint32_t getN() const {
        return n;
    }
private:
    std::uint32_t m{};
    std::uint32_t n{};
    Tensile::DataType datatype{Tensile::DataType::Float};
};

class LayerNormSolutionLibrary : public ExtOpLibrary {
public:
    static constexpr char opName[] = "LayerNorm";

    ~LayerNormSolutionLibrary() override {}
    void addSolution(LayerNormSolution &sol) {
        solutions.push_back(std::make_shared<LayerNormSolution>(sol));
    }

    std::string type() const override {
        return "LayerNormSolutionLibrary";
    }

    std::string description() const override {
        return "LayerNormSolutionLibrary";
    }

    std::shared_ptr<LayerNormSolution> findBestSolution(
        const LayerNormProblem &problem,
        const Tensile::Hardware &hardware,
        double* fitness = nullptr) const {
        auto bestSolIter = std::lower_bound(begin(solutions), end(solutions), problem.getN(), [](const auto &it, auto v) {
            return it->getLimit() < v;
        });

        return *bestSolIter;
    }

    void sortSolutions() {
        std::sort(begin(solutions), end(solutions), [](const auto &lhs, const auto &rhs) {
            return lhs->getLimit() < rhs->getLimit();
        });
    }

private:
    Tensile::SolutionVector<LayerNormSolution> solutions;
};


class ExtOpMasterLibrary {
public:
    using ExtOpLibraryPtr = std::unique_ptr<ExtOpLibrary>;
    explicit ExtOpMasterLibrary(const std::string &libPath)
    : libPath(libPath) {
        libDir = std::string(dirname(&this->libPath[0]));
        load(libPath);
    }

    const ExtOpLibraryPtr &getLibrary(const std::string &archName, const std::string &opName) const {
        return libraries.at(archName).at(opName);
    }

    const std::string getLibraryPath() const {
        return libPath;
    }

    const std::string getLibraryFolder() const {
        return libDir;
    }

private:
    bool load(const std::string &libPath) {
        msgpack::object_handle handle;

        std::ifstream ifs(libPath, std::ios::in | std::ios::binary);

        if (!ifs.is_open()) {
            throw std::runtime_error("Invalid ext op lib path");
        }

        msgpack::unpacker unpacker;
        bool finished{};
        constexpr std::size_t bufferSize = 1 << 16;

        while (!finished && !ifs.fail()) {
            unpacker.reserve_buffer(bufferSize);
            ifs.read(unpacker.buffer(), bufferSize);
            unpacker.buffer_consumed(ifs.gcount());
            finished = unpacker.next(handle);
        }

        if (!finished) {
            throw std::runtime_error("Unexpected EOF!");
        }

        msgpack::object root = handle.get();
        std::unordered_map<std::string, msgpack::object> objMap;
        Tensile::Serialization::objectToMap(root, objMap);

        for (auto &archObj : objMap) {
            libraries.emplace(archObj.first, std::map<std::string, ExtOpLibraryPtr>());
            std::unordered_map<std::string, msgpack::object> opMap;
            Tensile::Serialization::objectToMap(archObj.second, opMap);

            for (auto &opLib : opMap) {
                auto &rawKernels = opLib.second;

                if (rawKernels.type != msgpack::type::ARRAY) {
                    throw std::runtime_error("Invalid ext op lib format");
                }

                const auto numKernels = rawKernels.via.array.size;

                if (opLib.first == "Softmax") {
                    libraries.at(archObj.first).emplace(opLib.first, std::make_unique<SoftmaxSolutionLibrary>());
                    auto &lib = libraries.at(archObj.first).at(opLib.first)->as<SoftmaxSolutionLibrary>();

                    for (uint32_t i = 0; i < numKernels; ++i) {
                        auto &rawKernel = rawKernels.via.array.ptr[i];
                        SoftmaxSolution solution;
                        Tensile::Serialization::MessagePackInput msgInput(rawKernel);
                        Tensile::Serialization::MappingTraits<SoftmaxSolution,
                            Tensile::Serialization::MessagePackInput>::mapping(msgInput, solution);

                        lib.addSolution(solution);
                    }

                    lib.sortSolutions();
                } else if (opLib.first == "LayerNorm"){
                    libraries.at(archObj.first).emplace(opLib.first, std::make_unique<LayerNormSolutionLibrary>());
                    auto &lib = libraries.at(archObj.first).at(opLib.first)->as<LayerNormSolutionLibrary>();

                    for (uint32_t i = 0; i < numKernels; ++i) {
                        auto &rawKernel = rawKernels.via.array.ptr[i];
                        LayerNormSolution solution;
                        Tensile::Serialization::MessagePackInput msgInput(rawKernel);
                        Tensile::Serialization::MappingTraits<LayerNormSolution,
                            Tensile::Serialization::MessagePackInput>::mapping(msgInput, solution);

                        lib.addSolution(solution);
                    }

                    lib.sortSolutions();
                }
            }
        }

        return true;
    }

private:
    std::map<std::string, std::map<std::string, ExtOpLibraryPtr>> libraries;
    std::string libPath;
    std::string libDir;
};
