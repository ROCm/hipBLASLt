#pragma once

#include "auxiliary.hpp"
#include "tensile_host.hpp"
#include <Tensile/DataTypes.hpp>

#include <map>
#include <string>
#include <vector>

class OverrideSingleton
{
public:
    std::string file_path;
    bool        env_mode = false;

    static OverrideSingleton& getInstance()
    {
        static OverrideSingleton gInstance;
        return gInstance;
    }

    // copy contructor
    OverrideSingleton(const OverrideSingleton&) = delete;
    // assignment operator
    OverrideSingleton& operator=(const OverrideSingleton&) = delete;

private:
    OverrideSingleton()
    {
        char* Env = getenv("HIPBLASLT_TUNING_OVERRIDE_FILE");
        if(Env)
        {
            file_path = Env;
            env_mode  = true;
        }
    }

    ~OverrideSingleton() {}
};

namespace TensileLite
{
    class ProblemOverride
    {
    public:
        ProblemOverride();
        ProblemOverride(bool     transA,
                        bool     transB,
                        DataType inputType,
                        DataType computeType,
                        DataType outputType,
                        size_t   m,
                        size_t   n,
                        size_t   k,
                        size_t   batchSize);
        ProblemOverride(const ProblemOverride& problem);

        inline bool transA() const
        {
            return m_transA;
        }
        inline bool transB() const
        {
            return m_transB;
        }
        inline DataType inputType() const
        {
            return m_inputType;
        }
        inline DataType computeType() const
        {
            return m_computeType;
        }
        inline DataType outputType() const
        {
            return m_outputType;
        }
        inline size_t m() const
        {
            return m_m;
        }
        inline size_t n() const
        {
            return m_n;
        }
        inline size_t k() const
        {
            return m_k;
        }
        inline size_t batchSize() const
        {
            return m_batchSize;
        }

    private:
        bool     m_transA;
        bool     m_transB;
        DataType m_inputType;
        DataType m_computeType;
        DataType m_outputType;
        size_t   m_m;
        size_t   m_n;
        size_t   m_k;
        size_t   m_batchSize;
    };

    std::pair<ProblemOverride, int> problemFromEntries(const std::vector<std::string>& entries);

    std::multimap<ProblemOverride, int> getContractionProblemsFromFile(const std::string& path);

    template <>
    struct Comparison<ProblemOverride>
    {
        enum
        {
            implemented = true
        };

        static int compare(ProblemOverride const& lhs, ProblemOverride const& rhs)
        {
            return LexicographicCompare(lhs.transA(),
                                        rhs.transA(),
                                        lhs.transB(),
                                        rhs.transB(),
                                        lhs.inputType(),
                                        rhs.inputType(),
                                        lhs.computeType(),
                                        rhs.computeType(),
                                        lhs.outputType(),
                                        rhs.outputType(),
                                        lhs.m(),
                                        rhs.m(),
                                        lhs.n(),
                                        rhs.n(),
                                        lhs.k(),
                                        rhs.k(),
                                        lhs.batchSize(),
                                        rhs.batchSize());
        }
    };

} // namespace Tensile

namespace std
{
    template <>
    struct hash<TensileLite::ProblemOverride>
    {
        inline size_t operator()(TensileLite::ProblemOverride const& po) const
        {
            return TensileLite::hash_combine(po.transA(),
                                             po.transB(),
                                             po.inputType(),
                                             po.computeType(),
                                             po.outputType(),
                                             po.m(),
                                             po.n(),
                                             po.k(),
                                             po.batchSize());
        }
    };
} // namespace std
