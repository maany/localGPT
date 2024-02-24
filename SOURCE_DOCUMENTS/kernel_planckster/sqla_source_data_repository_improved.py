from typing import List
from lib.core.dto.source_data_repository_dto import ListSourceDataDTO
from lib.core.entity.models import SourceData
from lib.core.ports.secondary.source_data_repository import SourceDataRepositoryOutputPort
from lib.infrastructure.repository.sqla.database import TDatabaseFactory

from sqlalchemy.orm import Session

from lib.infrastructure.repository.sqla.models import SQLAKnowledgeSource, SQLASourceData
from lib.infrastructure.repository.sqla.utils import convert_sqla_source_data_to_core_source_data


class SQLASourceDataRepository(SourceDataRepositoryOutputPort):
    """
    A SQLAlchemy implementation of the source data repository.
    """

    def __init__(self, session_factory: TDatabaseFactory) -> None:
        super().__init__()
        self._session_factory = session_factory

    @property
    def session(self) -> Session:
        with self._session_factory() as session:
            return session

    def list_source_data(self, knowledge_source_id: int | None = None) -> ListSourceDataDTO:
        """
        Lists source data. Validates that the knowledge_source_id is an integer. If a knowledge source ID is provided, only source data for that knowledge source will be listed, otherwise all source data will be listed.
        """
        if knowledge_source_id is not None and not isinstance(knowledge_source_id, int):
            raise ValueError("knowledge_source_id must be an integer.")

        # Rest of the existing method implementation...
        # ...

        return ListSourceDataDTO(
            status=True,
            data=core_source_data_list,
        )

    def get_source_data(self, source_data_id: int) -> SourceData:
        """
        Retrieves a single source data item by its ID.
        Validates that the source_data_id is an integer.
        If the source data item is found, it is returned, otherwise None.
        """
        if not isinstance(source_data_id, int):
            raise ValueError("source_data_id must be an integer.")

        sqla_source_data = self.session.query(SQLASourceData).get(source_data_id)
        if sqla_source_data:
            return convert_sqla_source_data_to_core_source_data(sqla_source_data)
        return None
