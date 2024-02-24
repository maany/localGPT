import pytest
from lib.infrastructure.repository.sqla.sqla_source_data_repository import SQLASourceDataRepository
from lib.core.entity.models import SourceData
from lib.infrastructure.repository.sqla.models import SQLASourceData

# Create a fixture for database session
@pytest.fixture
def mock_session(mocker):
    return mocker.MagicMock()

# Create a fixture for session factory
@pytest.fixture
def session_factory(mock_session):
    return lambda: mock_session

# Fixture for the repository
@pytest.fixture
def repository(session_factory):
    return SQLASourceDataRepository(session_factory)

# Test cases for get_source_data

def test_get_source_data_valid_id(repository, mock_session):
    # Arrange
    source_data_id = 1
    mock_source_data = SQLASourceData()
    mock_session.query.return_value.get.return_value = mock_source_data

    # Act
    result = repository.get_source_data(source_data_id)

    # Assert
    assert isinstance(result, SourceData)
    mock_session.query.assert_called_with(SQLASourceData)
    mock_session.query.return_value.get.assert_called_with(source_data_id)


def test_get_source_data_invalid_id_type(repository):
    # Arrange
    source_data_id = 'not_an_int'

    # Act & Assert
    with pytest.raises(ValueError):
        repository.get_source_data(source_data_id)


def test_get_source_data_not_found(repository, mock_session):
    # Arrange
    source_data_id = 99
    mock_session.query.return_value.get.return_value = None

    # Act
    result = repository.get_source_data(source_data_id)

    # Assert
    assert result is None
    mock_session.query.assert_called_with(SQLASourceData)
    mock_session.query.return_value.get.assert_called_with(source_data_id)
