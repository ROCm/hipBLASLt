/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocblaslt-auxiliary.h"
#include <Tensile/Serialization.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/msgpack/MessagePack.hpp>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <libgen.h>
#include <msgpack.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

class SoftmaxProblem;
class SoftmaxSolution;

class SoftmaxSolution : public Tensile::Solution
{
public:
    friend struct Tensile::Serialization::MappingTraits<SoftmaxSolution,
                                                        Tensile::Serialization::MessagePackInput>;

    using Problem = SoftmaxProblem;
    std::string name() const override
    {
        return kernelName;
    }

    std::string description() const override
    {
        std::stringstream ss;
        ss << "Softmax, (Datatype, tileM, tileN) = "
           << "(" << Tensile::ToString(datatype) << ", " << tileM << ", " << tileN << ")";
        return ss.str();
    }

    std::uint32_t getTileM() const
    {
        return tileM;
    }

    std::uint32_t getTileN() const
    {
        return tileN;
    }

    std::uint32_t getNumWorkitems() const
    {
        return numWorkitems;
    }

    std::string getCodeObjectPath() const
    {
        return coPath;
    }

    Tensile::DataType getDatatype() const
    {
        return datatype;
    }

private:
    std::size_t       tileM{};
    std::size_t       tileN{};
    std::size_t       numWorkitems{};
    std::string       coPath;
    std::string       kernelName;
    Tensile::DataType datatype;
};

template <typename IO>
struct Tensile::Serialization::MappingTraits<SoftmaxSolution, IO>
{
    using iot = IOTraits<IO>;
    static void mapping(IO& io, SoftmaxSolution& s)
    {
        iot::mapRequired(io, "func_name", s.kernelName);
        std::string datatypeStr;
        iot::mapRequired(io, "io_type", datatypeStr);

        if(datatypeStr == "S")
        {
            s.datatype = Tensile::DataType::Float;
        }
        else
        {
            throw std::runtime_error("Invalid datatype in ext op library");
        }

        iot::mapRequired(io, "num_rows", s.tileM);
        iot::mapRequired(io, "num_cols", s.tileN);
        iot::mapRequired(io, "num_workitems", s.numWorkitems);
        iot::mapRequired(io, "co_path", s.coPath);
    }

    const static bool flow = false;
};

class SoftmaxProblem : public Tensile::Problem
{
public:
    using Solution = SoftmaxSolution;
    SoftmaxProblem(uint32_t m, uint32_t n, Tensile::DataType datatype)
        : m(m)
        , n(n)
        , datatype(datatype)
    {
    }

    ~SoftmaxProblem() override {}

    std::string description() const override
    {
        std::stringstream ss;
        ss << "Softmax Problem(" << ToString(datatype) << ", " << m << ", " << n << ")";
        return ss.str();
    }

    std::uint32_t getM() const
    {
        return m;
    }

    std::uint32_t getN() const
    {
        return n;
    }

private:
    std::uint32_t     m{};
    std::uint32_t     n{};
    Tensile::DataType datatype{Tensile::DataType::Float};
};

struct ExtOpLibrary
{
    virtual ~ExtOpLibrary()                 = default;
    virtual std::string type() const        = 0;
    virtual std::string description() const = 0;

    template <typename T>
    T& as()
    {
        return dynamic_cast<T&>(*this);
    }

    template <typename T>
    const T& as() const
    {
        return dynamic_cast<T&>(*this);
    }
};

class SoftmaxSolutionLibrary : public ExtOpLibrary
{
public:
    static constexpr char opName[] = "Softmax";

    ~SoftmaxSolutionLibrary() override {}
    void addSolution(SoftmaxSolution& sol)
    {
        solutions.push_back(std::make_shared<SoftmaxSolution>(sol));
    }

    std::string type() const override
    {
        return "SoftmaxSolutionLibrary";
    }

    std::string description() const override
    {
        return "SoftmaxSolutionLibrary";
    }

    std::shared_ptr<SoftmaxSolution> findBestSolution(const SoftmaxProblem&    problem,
                                                      const Tensile::Hardware& hardware,
                                                      double* fitness = nullptr) const
    {
        auto bestSolIter = std::lower_bound(
            begin(solutions), end(solutions), problem.getN(), [](const auto& it, auto v) {
                return it->getTileN() < v;
            });

        return *bestSolIter;
    }

    void sortSolutions()
    {
        std::sort(begin(solutions), end(solutions), [](const auto& lhs, const auto& rhs) {
            return lhs->getTileN() < rhs->getTileN();
        });
    }

private:
    Tensile::SolutionVector<SoftmaxSolution> solutions;
};

class LayerNormProblem;
class LayerNormSolution;

class LayerNormSolution : public Tensile::Solution
{
public:
    friend struct Tensile::Serialization::MappingTraits<LayerNormSolution,
                                                        Tensile::Serialization::MessagePackInput>;

    using Problem = LayerNormProblem;
    std::string name() const override
    {
        return kernelName;
    }

    std::string description() const override
    {
        std::stringstream ss;
        ss << "LayerNorm, (Datatype) = "
           << "(" << Tensile::ToString(datatype) << ")";
        return ss.str();
    }

    std::uint32_t getNumWorkitems() const
    {
        return numWorkitems;
    }

    std::uint32_t getLimit() const
    {
        return limit;
    }

    std::string getCodeObjectPath() const
    {
        return coPath;
    }

    Tensile::DataType getDatatype() const
    {
        return datatype;
    }

private:
    std::size_t       numWorkitems{};
    std::size_t       limit;
    std::string       coPath;
    std::string       kernelName;
    Tensile::DataType datatype;
};

template <typename IO>
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

        if(datatypeStr == "S")
        {
            s.datatype = Tensile::DataType::Float;
        }
        else
        {
            throw std::runtime_error("Invalid datatype in ext op library");
        }
    }

    const static bool flow = false;
};

class LayerNormProblem : public Tensile::Problem
{
public:
    using Solution = LayerNormSolution;
    LayerNormProblem(uint32_t m, uint32_t n, Tensile::DataType datatype)
        : m(m)
        , n(n)
        , datatype(datatype)
    {
    }

    ~LayerNormProblem() override {}

    std::string description() const override
    {
        std::stringstream ss;
        ss << "LayerNorm Problem(" << ToString(datatype) << ", " << m << ", " << n << ")";
        return ss.str();
    }

    std::uint32_t getM() const
    {
        return m;
    }

    std::uint32_t getN() const
    {
        return n;
    }

private:
    std::uint32_t     m{};
    std::uint32_t     n{};
    Tensile::DataType datatype{Tensile::DataType::Float};
};

class LayerNormSolutionLibrary : public ExtOpLibrary
{
public:
    static constexpr char opName[] = "LayerNorm";

    ~LayerNormSolutionLibrary() override {}
    void addSolution(LayerNormSolution& sol)
    {
        solutions.push_back(std::make_shared<LayerNormSolution>(sol));
    }

    std::string type() const override
    {
        return "LayerNormSolutionLibrary";
    }

    std::string description() const override
    {
        return "LayerNormSolutionLibrary";
    }

    std::shared_ptr<LayerNormSolution> findBestSolution(const LayerNormProblem&  problem,
                                                        const Tensile::Hardware& hardware,
                                                        double* fitness = nullptr) const
    {
        auto bestSolIter = std::lower_bound(
            begin(solutions), end(solutions), problem.getN(), [](const auto& it, auto v) {
                return it->getLimit() < v;
            });

        return *bestSolIter;
    }

    void sortSolutions()
    {
        std::sort(begin(solutions), end(solutions), [](const auto& lhs, const auto& rhs) {
            return lhs->getLimit() < rhs->getLimit();
        });
    }

private:
    Tensile::SolutionVector<LayerNormSolution> solutions;
};

class AMaxProblem;
class AMaxSolution;

class AMaxSolution : public Tensile::Solution
{
public:
    friend struct Tensile::Serialization::MappingTraits<AMaxSolution,
                                                        Tensile::Serialization::MessagePackInput>;

    using Problem = AMaxProblem;
    std::string name() const override
    {
        return kernelName;
    }

    std::string description() const override
    {
        std::stringstream ss;
        ss << "AMax, (Datatype, outDatatype) = "
           << "(" << Tensile::ToString(datatype) << ", " << Tensile::ToString(outDatatype) << ")";
        return ss.str();
    }

    std::uint32_t getNumWorkitems() const
    {
        return numWorkitems;
    }

    std::string getCodeObjectPath() const
    {
        return coPath;
    }

    Tensile::DataType getDatatype() const
    {
        return datatype;
    }

    Tensile::DataType getOutDatatype() const
    {
        return outDatatype;
    }

    Tensile::DataType getScaleDatatype() const
    {
        return scaleDatatype;
    }

    bool getIsScale() const
    {
        return isScale;
    }

private:
    std::size_t       numWorkitems{};
    std::string       coPath;
    std::string       kernelName;
    Tensile::DataType datatype;
    Tensile::DataType outDatatype;
    Tensile::DataType scaleDatatype;
    bool              isScale;
};

template <typename IO>
struct Tensile::Serialization::MappingTraits<AMaxSolution, IO>
{
    using iot = IOTraits<IO>;
    static void mapping(IO& io, AMaxSolution& s)
    {
        std::string datatypeStr;
        std::string outDatatypeStr;
        std::string scaleDatatypeStr;

        // add co_path, remove arch and op
        iot::mapRequired(io, "co_path", s.coPath);
        iot::mapRequired(io, "func_name", s.kernelName);
        iot::mapRequired(io, "io_type", datatypeStr);
        iot::mapRequired(io, "o_type", outDatatypeStr);
        iot::mapRequired(io, "scale_type", scaleDatatypeStr);
        iot::mapRequired(io, "num_workitems", s.numWorkitems);
        iot::mapRequired(io, "is_scale", s.isScale);

        if(datatypeStr == "S")
        {
            s.datatype = Tensile::DataType::Float;
        }
        else if(datatypeStr == "H")
        {
            s.datatype = Tensile::DataType::Half;
        }
        else
        {
            throw std::runtime_error("Invalid datatype in ext op library");
        }

        if(outDatatypeStr == "S")
        {
            s.outDatatype = Tensile::DataType::Float;
        }
        else if(outDatatypeStr == "H")
        {
            s.outDatatype = Tensile::DataType::Half;
        }
        else
        {
            throw std::runtime_error("Invalid datatype in ext op library");
        }

        if(scaleDatatypeStr == "F8")
        {
            s.scaleDatatype = Tensile::DataType::Float8;
        }
        else if(scaleDatatypeStr == "B8")
        {
            s.scaleDatatype = Tensile::DataType::BFloat8;
        }
        else
        {
            throw std::runtime_error("Invalid datatype in ext op library");
        }
    }

    const static bool flow = false;
};

class AMaxProblem : public Tensile::Problem
{
public:
    using Solution = AMaxSolution;
    AMaxProblem(uint32_t length, Tensile::DataType datatype, Tensile::DataType outDatatype)
        : length(length)
        , datatype(datatype)
        , outDatatype(outDatatype)
    {
    }
    AMaxProblem(uint32_t          length,
                Tensile::DataType datatype,
                Tensile::DataType outDatatype,
                Tensile::DataType scaleDatatype,
                bool              isScale)
        : length(length)
        , datatype(datatype)
        , outDatatype(outDatatype)
        , scaleDatatype(scaleDatatype)
        , isScale(isScale)
    {
    }

    ~AMaxProblem() override {}

    std::string description() const override
    {
        std::stringstream ss;
        ss << "AMax Problem(" << ToString(datatype) << ", " << ToString(outDatatype) << ", "
           << length << ")";
        return ss.str();
    }

    std::uint32_t getLength() const
    {
        return length;
    }

    Tensile::DataType getDatatype() const
    {
        return datatype;
    }

    Tensile::DataType getOutDatatype() const
    {
        return outDatatype;
    }

    Tensile::DataType getScaleDatatype() const
    {
        return scaleDatatype;
    }

    bool getIsScale() const
    {
        return isScale;
    }

private:
    std::uint32_t     length{};
    Tensile::DataType datatype{Tensile::DataType::Float};
    Tensile::DataType outDatatype{Tensile::DataType::Float};
    Tensile::DataType scaleDatatype{Tensile::DataType::Float8};
    bool              isScale = false;
};

class AMaxSolutionLibrary : public ExtOpLibrary
{
public:
    static constexpr char opName[] = "AMax";

    ~AMaxSolutionLibrary() override {}
    void addSolution(AMaxSolution& sol)
    {
        solutions.push_back(std::make_shared<AMaxSolution>(sol));
    }

    std::string type() const override
    {
        return "AMaxSolutionLibrary";
    }

    std::string description() const override
    {
        return "AMaxSolutionLibrary";
    }

    std::shared_ptr<AMaxSolution> findBestSolution(const AMaxProblem&       prob,
                                                   const Tensile::Hardware& hardware,
                                                   double*                  fitness = nullptr) const
    {
        if(prob.getIsScale())
        {
            for(auto sol : solutions)
                if(sol->getOutDatatype() == prob.getOutDatatype()
                   && sol->getScaleDatatype() == prob.getScaleDatatype() && sol->getIsScale())
                    return sol;
        }
        else
        {
            for(auto sol : solutions)
                if(sol->getOutDatatype() == prob.getOutDatatype() && !sol->getIsScale())
                    return sol;
        }
        return nullptr;
    }

    void sortSolutions()
    {
        std::sort(begin(solutions), end(solutions), [](const auto& lhs, const auto& rhs) {
            return lhs->getOutDatatype() < rhs->getOutDatatype();
        });
    }

private:
    Tensile::SolutionVector<AMaxSolution> solutions;
};


class AMax2DProblem;
class AMax2DSolution;

class AMax2DSolution : public Tensile::Solution
{
public:
    friend struct Tensile::Serialization::MappingTraits<AMax2DSolution,
                                                        Tensile::Serialization::MessagePackInput>;

    using Problem = AMax2DProblem;
    std::string name() const override
    {
        return kernelName;
    }

    std::string description() const override
    {
        std::stringstream ss;
        ss << "AMax2D, (Datatype, outDatatype) = "
           << "(" << Tensile::ToString(datatype) << ", " << Tensile::ToString(outDatatype) << ")";
        return ss.str();
    }

    std::uint32_t getNumWorkitems() const
    {
        return numWorkitems;
    }

    std::string getCodeObjectPath() const
    {
        return coPath;
    }

    Tensile::DataType getDatatype() const
    {
        return datatype;
    }

    Tensile::DataType getOutDatatype() const
    {
        return outDatatype;
    }

    Tensile::DataType getScaleDatatype() const
    {
        return scaleDatatype;
    }

    bool getIsScale() const
    {
        return isScale;
    }

private:
    std::size_t       numWorkitems{};
    std::string       coPath;
    std::string       kernelName;
    Tensile::DataType datatype;
    Tensile::DataType outDatatype;
    Tensile::DataType scaleDatatype;
    bool              isScale;
};

template <typename IO>
struct Tensile::Serialization::MappingTraits<AMax2DSolution, IO>
{
    using iot = IOTraits<IO>;
    static void mapping(IO& io, AMax2DSolution& s)
    {
        std::string datatypeStr;
        std::string outDatatypeStr;
        std::string scaleDatatypeStr;

        // add co_path, remove arch and op
        iot::mapRequired(io, "co_path", s.coPath);
        iot::mapRequired(io, "func_name", s.kernelName);
        iot::mapRequired(io, "io_type", datatypeStr);
        iot::mapRequired(io, "o_type", outDatatypeStr);
        iot::mapRequired(io, "scale_type", scaleDatatypeStr);
        iot::mapRequired(io, "num_workitems", s.numWorkitems);
        iot::mapRequired(io, "is_scale", s.isScale);

        if(datatypeStr == "S")
        {
            s.datatype = Tensile::DataType::Float;
        }
        else if(datatypeStr == "H")
        {
            s.datatype = Tensile::DataType::Half;
        }
        else
        {
            throw std::runtime_error("Invalid datatype in ext op library");
        }

        if(outDatatypeStr == "S")
        {
            s.outDatatype = Tensile::DataType::Float;
        }
        else if(outDatatypeStr == "H")
        {
            s.outDatatype = Tensile::DataType::Half;
        }
        else
        {
            throw std::runtime_error("Invalid datatype in ext op library");
        }

        if(scaleDatatypeStr == "F8")
        {
            s.scaleDatatype = Tensile::DataType::Float8;
        }
        else if(scaleDatatypeStr == "B8")
        {
            s.scaleDatatype = Tensile::DataType::BFloat8;
        }
        else
        {
            throw std::runtime_error("Invalid datatype in ext op library");
        }
    }

    const static bool flow = false;
};

class AMax2DProblem : public Tensile::Problem
{
public:
    using Solution = AMax2DSolution;
    AMax2DProblem(uint32_t length, Tensile::DataType datatype, Tensile::DataType outDatatype)
        : length(length)
        , datatype(datatype)
        , outDatatype(outDatatype)
    {
    }
    AMax2DProblem(uint32_t          length,
                  Tensile::DataType datatype,
                  Tensile::DataType outDatatype,
                  Tensile::DataType scaleDatatype,
                  bool              isScale)
        : length(length)
        , datatype(datatype)
        , outDatatype(outDatatype)
        , scaleDatatype(scaleDatatype)
        , isScale(isScale)
    {
    }

    ~AMax2DProblem() override {}

    std::string description() const override
    {
        std::stringstream ss;
        ss << "AMax2D Problem(" << ToString(datatype) << ", " << ToString(outDatatype) << ", "
           << length << ")";
        return ss.str();
    }

    std::uint32_t getLength() const
    {
        return length;
    }

    Tensile::DataType getDatatype() const
    {
        return datatype;
    }

    Tensile::DataType getOutDatatype() const
    {
        return outDatatype;
    }

    Tensile::DataType getScaleDatatype() const
    {
        return scaleDatatype;
    }

    bool getIsScale() const
    {
        return isScale;
    }

private:
    std::uint32_t     length{};
    Tensile::DataType datatype{Tensile::DataType::Float};
    Tensile::DataType outDatatype{Tensile::DataType::Float};
    Tensile::DataType scaleDatatype{Tensile::DataType::Float8};
    bool              isScale = false;
};

class AMax2DSolutionLibrary : public ExtOpLibrary
{
public:
    static constexpr char opName[] = "AMax2D";

    ~AMax2DSolutionLibrary() override {}
    void addSolution(AMax2DSolution& sol)
    {
        solutions.push_back(std::make_shared<AMax2DSolution>(sol));
    }

    std::string type() const override
    {
        return "AMax2DSolutionLibrary";
    }

    std::string description() const override
    {
        return "AMax2DSolutionLibrary";
    }

    std::shared_ptr<AMax2DSolution> findBestSolution(const AMax2DProblem&       prob,
                                                     const Tensile::Hardware& hardware,
                                                     double*                  fitness = nullptr) const
    {
        if(prob.getIsScale())
        {
            for(auto sol : solutions)
                if(sol->getOutDatatype() == prob.getOutDatatype()
                   && sol->getScaleDatatype() == prob.getScaleDatatype() && sol->getIsScale())
                    return sol;
        }
        else
        {
            for(auto sol : solutions)
                if(sol->getOutDatatype() == prob.getOutDatatype() && !sol->getIsScale())
                    return sol;
        }
        return nullptr;
    }

    void sortSolutions()
    {
        std::sort(begin(solutions), end(solutions), [](const auto& lhs, const auto& rhs) {
            return lhs->getOutDatatype() < rhs->getOutDatatype();
        });
    }

private:
    Tensile::SolutionVector<AMax2DSolution> solutions;
};


class ExtOpMasterLibrary
{
public:
    using ExtOpLibraryPtr = std::unique_ptr<ExtOpLibrary>;
    explicit ExtOpMasterLibrary(const std::string& libPath)
        : libPath(libPath)
    {
        libDir = std::string(dirname(&this->libPath[0]));
        load(libPath);
    }

    const ExtOpLibraryPtr& getLibrary(const std::string& archName,
                                      const std::string& opName,
                                      const std::string& typeName) const
    {
        return libraries.at(archName).at(opName).at(typeName);
    }

    const std::string getLibraryPath() const
    {
        return libPath;
    }

    const std::string getLibraryFolder() const
    {
        return libDir;
    }

private:
    bool load(const std::string& libPath)
    {
        msgpack::object_handle handle;

        std::ifstream ifs(libPath, std::ios::in | std::ios::binary);

        if(!ifs.is_open())
        {
            throw std::runtime_error("Invalid ext op lib path");
        }

        msgpack::unpacker     unpacker;
        bool                  finished{};
        constexpr std::size_t bufferSize = 1 << 16;

        while(!finished && !ifs.fail())
        {
            unpacker.reserve_buffer(bufferSize);
            ifs.read(unpacker.buffer(), bufferSize);
            unpacker.buffer_consumed(ifs.gcount());
            finished = unpacker.next(handle);
        }

        if(!finished)
        {
            throw std::runtime_error("Unexpected EOF!");
        }

        msgpack::object                                  root = handle.get();
        std::unordered_map<std::string, msgpack::object> objMap;
        Tensile::Serialization::objectToMap(root, objMap);

        for(auto& archObj : objMap)
        {
            libraries.emplace(archObj.first,
                              std::map<std::string, std::map<std::string, ExtOpLibraryPtr>>());

            std::unordered_map<std::string, msgpack::object> opMap;
            Tensile::Serialization::objectToMap(archObj.second, opMap);

            for(auto& opObj : opMap)
            {
                libraries.at(archObj.first)
                    .emplace(opObj.first, std::map<std::string, ExtOpLibraryPtr>());

                std::unordered_map<std::string, msgpack::object> typeMap;
                Tensile::Serialization::objectToMap(opObj.second, typeMap);

                for(auto& typeLib : typeMap)
                {
                    auto& rawKernels = typeLib.second;

                    if(rawKernels.type != msgpack::type::ARRAY)
                    {
                        throw std::runtime_error("Invalid ext op lib format");
                    }

                    const auto numKernels = rawKernels.via.array.size;

                    if(opObj.first == "Softmax")
                    {
                        libraries.at(archObj.first)
                            .at(opObj.first)
                            .emplace(typeLib.first, std::make_unique<SoftmaxSolutionLibrary>());
                        auto& lib = libraries.at(archObj.first)
                                        .at(opObj.first)
                                        .at(typeLib.first)
                                        ->as<SoftmaxSolutionLibrary>();

                        for(uint32_t i = 0; i < numKernels; ++i)
                        {
                            auto&           rawKernel = rawKernels.via.array.ptr[i];
                            SoftmaxSolution solution;
                            Tensile::Serialization::MessagePackInput msgInput(rawKernel);
                            Tensile::Serialization::MappingTraits<
                                SoftmaxSolution,
                                Tensile::Serialization::MessagePackInput>::mapping(msgInput,
                                                                                   solution);

                            lib.addSolution(solution);
                        }

                        lib.sortSolutions();
                    }
                    else if(opObj.first == "LayerNorm")
                    {
                        libraries.at(archObj.first)
                            .at(opObj.first)
                            .emplace(typeLib.first, std::make_unique<LayerNormSolutionLibrary>());
                        auto& lib = libraries.at(archObj.first)
                                        .at(opObj.first)
                                        .at(typeLib.first)
                                        ->as<LayerNormSolutionLibrary>();

                        for(uint32_t i = 0; i < numKernels; ++i)
                        {
                            auto&             rawKernel = rawKernels.via.array.ptr[i];
                            LayerNormSolution solution;
                            Tensile::Serialization::MessagePackInput msgInput(rawKernel);
                            Tensile::Serialization::MappingTraits<
                                LayerNormSolution,
                                Tensile::Serialization::MessagePackInput>::mapping(msgInput,
                                                                                   solution);

                            lib.addSolution(solution);
                        }

                        lib.sortSolutions();
                    }
                    else if(opObj.first == "AMax")
                    {
                        libraries.at(archObj.first)
                            .at(opObj.first)
                            .emplace(typeLib.first, std::make_unique<AMaxSolutionLibrary>());
                        auto& lib = libraries.at(archObj.first)
                                        .at(opObj.first)
                                        .at(typeLib.first)
                                        ->as<AMaxSolutionLibrary>();

                        for(uint32_t i = 0; i < numKernels; ++i)
                        {
                            auto&        rawKernel = rawKernels.via.array.ptr[i];
                            AMaxSolution solution;
                            Tensile::Serialization::MessagePackInput msgInput(rawKernel);
                            Tensile::Serialization::MappingTraits<
                                AMaxSolution,
                                Tensile::Serialization::MessagePackInput>::mapping(msgInput,
                                                                                   solution);

                            lib.addSolution(solution);
                        }

                        lib.sortSolutions();
                    }
                    else if(opObj.first == "AMax2D")
                    {
                        libraries.at(archObj.first)
                            .at(opObj.first)
                            .emplace(typeLib.first, std::make_unique<AMax2DSolutionLibrary>());
                        auto& lib = libraries.at(archObj.first)
                                        .at(opObj.first)
                                        .at(typeLib.first)
                                        ->as<AMax2DSolutionLibrary>();

                        for(uint32_t i = 0; i < numKernels; ++i)
                        {
                            auto&        rawKernel = rawKernels.via.array.ptr[i];
                            AMax2DSolution solution;
                            Tensile::Serialization::MessagePackInput msgInput(rawKernel);
                            Tensile::Serialization::MappingTraits<
                                AMax2DSolution,
                                Tensile::Serialization::MessagePackInput>::mapping(msgInput,
                                                                                   solution);

                            lib.addSolution(solution);
                        }

                        lib.sortSolutions();
                    }
                }
            }
        }

        return true;
    }

private:
    std::map<std::string, std::map<std::string, std::map<std::string, ExtOpLibraryPtr>>> libraries;
    std::string                                                                          libPath;
    std::string                                                                          libDir;
};
