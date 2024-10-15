#include "UserDrivenTuningParser.hpp"


#include <fstream>
#include <sstream>
#include <utility>

Tensile::DataType roc2TensileType(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f32:
    case rocblaslt_compute_f32_fast_xf32:
    case rocblaslt_compute_f32_fast_f16:
    case rocblaslt_compute_f32_fast_bf16:
    case rocblaslt_compute_f32_fast_f8_fnuz:
    case rocblaslt_compute_f32_fast_bf8_fnuz:
    case rocblaslt_compute_f32_fast_f8bf8_fnuz:
    case rocblaslt_compute_f32_fast_bf8f8_fnuz:
#ifdef ROCM_USE_FLOAT8
    case rocblaslt_compute_f32_fast_f8_ocp:
    case rocblaslt_compute_f32_fast_bf8_ocp:
    case rocblaslt_compute_f32_fast_f8bf8_ocp:
    case rocblaslt_compute_f32_fast_bf8f8_ocp:
#endif
        return Tensile::DataType::Float;
    case rocblaslt_compute_f64:
        return Tensile::DataType::Double;
    case rocblaslt_compute_i32:
        return Tensile::DataType::Int32;
    case rocblaslt_compute_f16:
        return Tensile::DataType::Half;
    default:
        throw std::runtime_error("Unsupported type.");
    }
    return Tensile::DataType::None;
}

namespace Tensile
{

    std::multimap<ProblemOverride, int>
        getContractionProblemsFromFile(const std::string& path)
    {
        
        static std::multimap<ProblemOverride, int> m_override;
        
        if (m_override.size() == 0){
            
            std::ifstream file_read(path);
            std::string   line, entry;

            const auto verion           = "Git Version";
            const auto delim            = ',';
            const int  max_entries      = 37;

            while(std::getline(file_read, line))
            {
                // Ignore lines without delimiter
                line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));

                if(line.find(delim) != std::string::npos && line.find(verion) == std::string::npos)
                {
                    std::vector<std::string> entries{};
                    entries.reserve(max_entries);

                    std::stringstream line_ss(line);
                    while(getline(line_ss, entry, delim))
                    {
                        entries.push_back(entry);
                    }

                    auto problemSolution = problemFromEntries(entries);
                    
                    if(problemSolution.second > 0)
                    {
                        bool duplicated_entry = false;
                        auto sol_iter = m_override.equal_range(problemSolution.first);
                        for (auto sol_idx = sol_iter.first; 
                            !duplicated_entry && sol_idx != sol_iter.second; 
                            sol_idx++)
                        {
                            if (sol_idx->second == problemSolution.second)
                            {
                                duplicated_entry = true;
                                sol_idx = m_override.erase(sol_idx);
                            }
                        }

                        m_override.insert(problemSolution);
                    }
                }
            }
        }
        
        
        return m_override;
    }
    
    
    std::pair<ProblemOverride, int>
        problemFromEntries(const std::vector<std::string>& entries)
    {
        
        const size_t entries_n = entries.size();
        if(entries_n != 37)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        //Expected format: transA,transB,batch_count,M,N,K,input_type,output_type,compute_type,solution_index
        bool transA = (entries[0] != "N");
        bool transB = (entries[1] != "N");

        size_t m, n, b, k;
        DataType inputType   = DataType::None;
        DataType outputType  = DataType::None;
        DataType computeType = DataType::None;

        int solution_idx = -1;

        try
        {
            
            // TODO: are any additional mapping parameters needed?

            b = std::stol(entries[3]);
            m = std::stol(entries[4]);
            n = std::stol(entries[5]);
            k = std::stol(entries[6]);
            inputType   = hipDataType_to_tensile_type(string_to_hip_datatype(entries[17]));
            outputType  = hipDataType_to_tensile_type(string_to_hip_datatype(entries[19]));
            computeType = hipDataType_to_tensile_type(string_to_hip_datatype(entries[21]));
            solution_idx = std::stoi(entries[34]);

        }
        catch(std::invalid_argument const& ex)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }
        catch(std::out_of_range const& ex)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        if(inputType == DataType::None || outputType == DataType::None
           || computeType == DataType::None)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        ProblemOverride  po(transA,
                            transB,
                            inputType,
                            computeType,
                            outputType,
                            m,
                            n,
                            k,
                            b);

        return std::make_pair(po, solution_idx);
        
    }

    
    ProblemOverride::ProblemOverride()
        : m_transA(false)
        , m_transB(false)
        , m_inputType(DataType::None)
        , m_computeType(DataType::None)
        , m_outputType(DataType::None)
        , m_m(0)
        , m_n(0)
        , m_k(0)
        , m_batchSize(0)
    {
    }

    
    ProblemOverride::ProblemOverride(bool     transA,
                                     bool     transB,
                                     DataType inputType,
                                     DataType computeType,
                                     DataType outputType,
                                     size_t   m,
                                     size_t   n,
                                     size_t   k,
                                     size_t   batchSize)
        : m_transA(transA)
        , m_transB(transB)
        , m_inputType(inputType)
        , m_computeType(computeType)
        , m_outputType(outputType)
        , m_m(m)
        , m_n(n)
        , m_k(k)
        , m_batchSize(batchSize)
    {
    }

    ProblemOverride::ProblemOverride(const RocblasltContractionProblem& problem)
    {
        if (problem.trans_a == HIPBLAS_OP_N)
            m_transA = false;
        else   
            m_transA = true;

        if (problem.trans_b == HIPBLAS_OP_N)
            m_transB = false;
        else   
            m_transB = true;
        m_inputType     = hipDataType_to_tensile_type(problem.a_type);
        m_computeType   = roc2TensileType(problem.compute_type);
        m_outputType    = hipDataType_to_tensile_type(problem.c_type);
        m_m             = problem.m;
        m_n             = problem.n;
        m_k             = problem.k;
        m_batchSize     = problem.batch_count;
        
    }

    ProblemOverride::ProblemOverride(const ContractionProblemGemm& problem)
    {
        
        m_transA        = problem.transA();
        m_transB        = problem.transB();
        m_inputType     = problem.a().dataType();
        m_computeType   = problem.computeInputType();
        m_outputType    = problem.c().dataType();
        m_m             = problem.freeSizeA(0);
        m_n             = problem.freeSizeB(0);
        m_k             = problem.boundSize(0);
        m_batchSize     = problem.batchSize(0);

    }
    
};
