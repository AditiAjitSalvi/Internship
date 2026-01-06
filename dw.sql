SELECT 
    A.ProgramNo,
    A.InstructionSrNo,
    A.Instruction,
    A.InstructionValue AS StationNumber,
    GetStationCriticality(A.InstructionValue) AS CriticalStatus
FROM AutoSequencePrograms A;

SELECT 
	A.seq_id,
    A.ProjectID                    AS project_id,
    A.WagonNumber                  AS wagon_no,
    A.InstructionSrNo              AS step_no,
    A.Instruction                  AS command,
    A.InstructionValue             AS station_no,
    COALESCE(S.CensorDistance, 0)  AS wait_sec,
    COALESCE(
        GetStationCriticality(A.InstructionValue),
        'UNKNOWN'
    )                              AS critical_status,
    COALESCE(S.Distance, 0)        AS total_time_sec,
    COALESCE(A.IsValid, 'TRUE')    AS valid
FROM AutoSequencePrograms A
LEFT JOIN StationMaster S
    ON  A.InstructionValue = S.StationNumber
    AND A.ProjectID = S.ProjectID;
    
    
    
    
   